# Copyright (c) 2025 ANTGROUP. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Linear Attention with Headwise Decay using CuTe DSL

This module implements chunkwise linear attention with per-head decay factors
for the NVIDIA Blackwell SM100 architecture using CUTE DSL.

The implementation supports:
- Chunkwise computation for improved GPU utilization
- Per-head decay coefficients for flexible dependency modeling
- Efficient state accumulation across chunks
- Input/output format: [Batch, Sequence, Heads, Dim]

To run this example:

.. code-block:: bash

    python examples/blackwell/linear_attn.py \\
      --batch_size 4 --seq_len 1024 --num_heads 8 --head_dim 64 \\
      --chunk_size 64 --decay 0.95

Mathematical formulation:

O_h(i) = (Q_i^T * (global_num + block_num)) / (Q_i^T * (global_denom + block_denom) + ε)

where:
- block_num(t) = λ_h * block_num(t-1) + K_t V_t^T
- block_denom(t) = λ_h * block_denom(t-1) + K_t
- global_state = λ_h^L * global_state + block_state (inter-chunk accumulation)
"""

import argparse
import math
import os
import sys
import time
from typing import Type, Tuple, List

import torch
import torch.nn.functional as F
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.cute.testing as testing
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32, Int64, Float32

class MaskEnum:
    """Enumeration for different mask types."""
    NONE = 0
    PADDING = 1
    CAUSAL = 2

class LinearAttentionChunkwise:
    """
    Chunkwise Linear Attention with Per-Head Decay using CuTe DSL
    
    Implements the Lightning Attention algorithm with headwise decay factors.
    Decomposes attention into intra-chunk (local) and inter-chunk (global) components.
    
    Args:
        chunk_size: Size of each attention chunk (default: 64)
        qk_acc_dtype: Accumulator data type for QK computation (default: Float32)
        kv_acc_dtype: Accumulator data type for PV computation (default: Float32)
        io_dtype: Input/output data type (default: Float16)
    """

    def __init__(
        self,
        chunk_size: int = 64,
        qk_acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        kv_acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        io_dtype: Type[cutlass.Numeric] = cutlass.BFloat16,
    ):
        self.chunk_size = chunk_size
        self.qk_acc_dtype = qk_acc_dtype
        self.kv_acc_dtype = kv_acc_dtype
        self.pv_acc_dtype = kv_acc_dtype
        self.io_dtype = io_dtype

        # Warp specialization
        self.num_load_warps = 1
        self.num_compute_warps = 4
        self.num_correction_warps = 4
        self.threads_per_warp = 32

        # MMA tile shapes
        # C: 64, choose chunk size as 64 for enough spaces to do double buffering
        # Q: (64, 128)
        # K: (64, 128)
        # V: (64, 128)
        # READ from input
        C, D = (64, 128)
        # (C, C, D)
        self.qk_mma_tiler = (C, C, D)  # (M, N, K)
        # (C, D, C)
        self.pv_mma_tiler = (C, D, C)  # (M, N, K)
        # (D, D, C)
        self.kv_mma_tiler = (D, D, C)  # (M, N, K)
        self.cta_tiler = self.qk_mma_tiler  # For simplicity, use same tiler

        # one-cta cluster shape
        self.cluster_shape_mnk = (1, 1, 1)
        self.decay_warp_ids = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.load_warp_id = 5
        self.epilogue_warp_id = 6
        self.empty_warp_id = 7

        SM100_TMEM_CAPACITY_COLS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLS

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.decay_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.epilogue_warp_id,
            )
        )

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )

        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_warp,
        )

        self.buffer_align_bytes = 1024

        self.tmem_p_offset = 0
        self.tmem_pm_offset = 0
        # Store hidden recurrent state s = k^tv
        self.tmem_s_offset = 0

        # Store states for this CTA (w/ head_idx, batch_idx Fixed)

        # Assume causal mask for this implementation
        self.mask_type = MaskEnum.CAUSAL

    @staticmethod
    def _plan_tmem_offsets(
        tiled_mma_qk,
        tile_shape_mnk_qk,
        tiled_mma_pv,
        tile_shape_mnk_pv,
        tiled_mma_kv,
        tile_shape_mnk_kv,
        acc_stages,
    ):
        """Compute TMEM offsets for various tensors used in the kernel."""
        SM100_TMEM_CAPACITY_COLS = 512
        BITS_PER_TMEM_COL = 32

        # (MMA, MMA_M, MMA_N)
        acc_shape_qk = tiled_mma_qk.partition_shape_C(tile_shape_mnk_qk[:2])
        # (MMA, MMA_M, MMA_N)
        tCtAccQK_fake = tiled_mma_qk.make_fragment_C(
            cute.append(acc_shape_qk, acc_stages)
        )
        num_qk_acc_cols = tcgen05.find_tmem_tensor_col_offset(tCtAccQK_fake)
        assert tile_shape_mnk_qk[1] * acc_stages == num_qk_acc_cols

        acc_shape_pv = tiled_mma_pv.partition_shape_C(tile_shape_mnk_pv[:2])
        tCtAccPV_fake = tiled_mma_pv.make_fragment_C(
            cute.append(acc_shape_pv, acc_stages)
        )
        num_pv_acc_cols = tcgen05.find_tmem_tensor_col_offset(tCtAccPV_fake)
        assert tile_shape_mnk_pv[1] * acc_stages == num_pv_acc_cols

        acc_shape_kv = tiled_mma_kv.partition_shape_C(tile_shape_mnk_kv[:2])
        tCtAccKV_fake = tiled_mma_kv.make_fragment_C(
            cute.append(acc_shape_kv, acc_stages)
        )
        num_kv_acc_cols = tcgen05.find_tmem_tensor_col_offset(tCtAccKV_fake)
        assert tile_shape_mnk_kv[1] * acc_stages == num_kv_acc_cols

        num_qk_acc_cols_offset = 0
        num_pv_acc_cols_offset = num_qk_acc_cols_offset + num_qk_acc_cols
        num_kv_acc_cols_offset = num_pv_acc_cols_offset + num_pv_acc_cols

        num_tmem_cols_total_tmp = num_kv_acc_cols_offset + num_kv_acc_cols
        # Turn num_tmem_cols_total to the nearest power of 2
        num_tmem_cols_total = 1
        while num_tmem_cols_total < num_tmem_cols_total_tmp:
            num_tmem_cols_total *= 2
        assert num_tmem_cols_total <= SM100_TMEM_CAPACITY_COLS

        return (
            num_qk_acc_cols_offset,
            num_pv_acc_cols_offset,
            num_kv_acc_cols_offset,
            num_tmem_cols_total,
        )

    def _setup_attributes(self):
        """Set up configurations and parameters for the linear attention kernel."""
        self.q_stage = 2
        self.k_stage = 2
        self.v_stage = 2
        self.o_stage = 2
        self.epi_stage = 2
        self.acc_stage = 1
        self.corr_stage = 1
        self.decay_stage = 2
        self.mma_mask_stage = 2

    def _compute_grid(
        self,
        o_shape: cute.Shape,
        chunk_size: int,
        ) -> cute.Shape:
        """Compute tile scheduler parameters based on the chunk size and MMA tiler."""
        return (
            # S / CHUNK
            # cute.ceil_div(o_shape[0], chunk_size),
            # For Loop to tile over chunk size,
            1,
            # B
            cute.size(o_shape[2][1]),
            # H
            cute.size(o_shape[2][0]),
        )

    @cute.jit
    def __call__(
        self,
        q_iter: cute.Pointer,
        k_iter: cute.Pointer,
        v_iter: cute.Pointer,
        o_iter: cute.Pointer,
        decay: cute.Pointer,
        problem_size: Tuple[Int32, Int32, Int32, Int32],  # (B, S, H, D)
        stream: cuda.CUstream,
    ):
        """
        Execute the Chunkwise Linear Attention operation on the provided tensors.
        
        Args:
            q_iter: Query tensor
            k_iter: Key tensor
            v_iter: Value tensor
            o_iter: Output tensor
            decay_iter: Per-head decay coefficients pointer [H]
            problem_size: (B, S, H, D) problem dimensions
            stream: CUDA stream
        """
        B,S,H,D = problem_size
        self.B, self.S, self.H, self.D = B, S, H, D

        # Setup attributes
        self._setup_attributes()

        self.qk_dim = D
        self.v_dim = D

        # Accum should be at least Float32 for numerical stability
        # At least 255 registers per threads.
        self.num_regs_st = self.qk_dim*self.v_dim*4 / 4

        self.cta_group = tcgen05.CtaGroup.ONE

        # It's ok since torch tensor is row major, hence we've layout=(B,S,H,D):(DHS, DH, D, 1).
        # Below are just permutation tricks to ease the later processing.
        q_layout = cute.make_layout(
            (S, D, (H,B)),
            stride=(D*H, 1, (D, D*H*S)),
        )
        q = cute.make_tensor(q_iter, q_layout)
        # (S, D, (H,B))
        k_layout = cute.make_layout(
            (S, D, (H,B)),
            stride=(D*H, 1, (D, D*H*S)),
        )
        k = cute.make_tensor(k_iter, k_layout)
        # (D, S, (H,B))
        v_layout = cute.make_layout(
            (D, S, (H,B)),
            stride=(1, D*H, (D, D*H*S)),
        )
        v = cute.make_tensor(v_iter, v_layout)
        # (S, D, (H,B))
        o_layout = cute.make_layout(
            (S, D, (H,B)),
            stride=(D*H, 1, (D, D*H*S)),
        )
        o = cute.make_tensor(o_iter, o_layout)

        # Hidden Final State
        fstate_layout = cute.make_layout(
            (D, D, (H, B)),
            stride=(1, D*H, (D, D*D*H)),
        )

        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.o_dtype = o.element_type

        self.q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()
        self.k_major_mode = utils.LayoutEnum.from_tensor(k).mma_major_mode()
        self.v_major_mode = utils.LayoutEnum.from_tensor(v).mma_major_mode()
        self.o_layout = utils.LayoutEnum.from_tensor(o)

        # self.q_major_mode = tcgen05.OperandMajorMode.K
        # self.k_major_mode = tcgen05.OperandMajorMode.K
        # self.v_major_mode = tcgen05.OperandMajorMode.MN

        if cutlass.const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of q is not supported")
        if cutlass.const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of k is not supported")
        if cutlass.const_expr(self.v_major_mode != tcgen05.OperandMajorMode.MN):
            raise RuntimeError("The layout of v is not supported")

        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            self.cta_group,
            self.qk_mma_tiler[:2],
        )
        kv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            self.k_major_mode,
            self.v_major_mode,
            self.kv_acc_dtype,
            self.cta_group,
            self.kv_mma_tiler[:2],
        )
        p_major_mode = tcgen05.OperandMajorMode.K
        # TODO: check p_source
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.pv_acc_dtype,
            self.cta_group,
            self.pv_mma_tiler[:2],
            tcgen05.OperandSource.TMEM,
        )

        (
            self.tmem_qk_cols_offset,
            self.tmem_pv_cols_offset,
            self.tmem_kv_cols_offset,
            self.tmem_total_cols,
        ) = self._plan_tmem_offsets(
            qk_tiled_mma,
            self.qk_mma_tiler,
            pv_tiled_mma,
            self.pv_mma_tiler,
            kv_tiled_mma,
            self.kv_mma_tiler,
            acc_stages=1,
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )

        # Output shape
        self.epi_tile = self.pv_mma_tiler[:2]

        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.q_dtype,
            self.q_stage,
        )
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.k_dtype,
            self.k_stage,
        )
        kt_smem_layout_staged = sm100_utils.make_smem_layout_a(
            kv_tiled_mma,
            self.kv_mma_tiler,
            self.k_dtype,
            self.k_stage,
        )
        p_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma,
            self.pv_mma_tiler,
            self.q_dtype,
            self.acc_stage,
        )
        v_smem_layout_staged = sm100_utils.make_smem_layout_b(
            kv_tiled_mma,
            self.kv_mma_tiler,
            self.v_dtype,
            self.v_stage,
        )
        o_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.o_dtype,
            self.o_layout,
            self.epi_tile,
            self.epi_stage,
        )

        # TMA operations
        # TODO: multicast check, (1,1,1) cluster indicates no multicast
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        # TMA load for Q
        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0,1,2])
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q,
            q_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # TMA load for K
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            k,
            k_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for V
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            v,
            v_smem_layout,
            self.kv_mma_tiler,
            kv_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # TMA store for O
        o_smem_layout = cute.select(o_smem_layout_staged, mode=[0, 1])
        tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_store_op,
            o,
            o_smem_layout,
            self.epi_tile,
        )

        q_copy_size = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        k_copy_size = cute.size_in_bytes(self.k_dtype, k_smem_layout)
        self.tma_copy_q_bytes = q_copy_size
        self.tma_copy_kv_bytes = k_copy_size        

        print(f"q_layout: {cute.pretty_str(q_layout)}")
        print(f"q: {cute.pretty_str(q)}")
        print(f"k_layout: {cute.pretty_str(k_layout)}")
        print(f"k: {cute.pretty_str(k)}")
        print(f"v_layout: {cute.pretty_str(v_layout)}")
        print(f"v: {cute.pretty_str(v)}")
        print(f"o_layout: {cute.pretty_str(o_layout)}")
        print(f"o: {cute.pretty_str(o)}")
        print(f"qk_tiled_mma: {cute.pretty_str(qk_tiled_mma)}")
        print(f"kv_tiled_mma: {cute.pretty_str(kv_tiled_mma)}")
        print(f"pv_tiled_mma: {cute.pretty_str(pv_tiled_mma)}")
        print(f"cluster_layout_vmnk: {cute.pretty_str(self.cluster_layout_vmnk)}")
        print(f"epi_tile: {cute.pretty_str(self.epi_tile)}")
        print(f"q_smem_layout: {cute.pretty_str(q_smem_layout)}")
        print(f"k_smem_layout: {cute.pretty_str(k_smem_layout)}")
        print(f"v_smem_layout: {cute.pretty_str(v_smem_layout)}")
        print(f"o_smem_layout: {cute.pretty_str(o_smem_layout)}")
        print(f"q_smem_layout_staged: {cute.pretty_str(q_smem_layout_staged)}")
        print(f"k_smem_layout_staged: {cute.pretty_str(k_smem_layout_staged)}")
        print(f"kt_smem_layout_staged: {cute.pretty_str(kt_smem_layout_staged)}")
        print(f"v_smem_layout_staged: {cute.pretty_str(v_smem_layout_staged)}")
        print(f"o_smem_layout_staged: {cute.pretty_str(o_smem_layout_staged)}")
        print(f"p_tmem_layout_staged: {cute.pretty_str(p_tmem_layout_staged)}")
        print(f"tma_atom_q: {cute.pretty_str(tma_atom_q)}")
        print(f"tma_atom_k: {cute.pretty_str(tma_atom_k)}")
        print(f"tma_atom_v: {cute.pretty_str(tma_atom_v)}")
        print(f"tma_atom_o: {cute.pretty_str(tma_atom_o)}")
        print(f"tma_tensor_q: {cute.pretty_str(tma_tensor_q)}")
        print(f"tma_tensor_k: {cute.pretty_str(tma_tensor_k)}")
        print(f"tma_tensor_v: {cute.pretty_str(tma_tensor_v)}")
        print(f"tma_tensor_o: {cute.pretty_str(tma_tensor_o)}")
        print(f"q_copy_size: {q_copy_size}")
        print(f"k_copy_size: {k_copy_size}")
        # Shared storage structure

        @cute.struct
        class SharedStorage:
            # Pipeline barriers
            # Inputs
            load_q_mbar_ptr: cute.struct.MemRange[Int64, self.q_stage * 2] # type: ignore
            load_k_mbar_ptr: cute.struct.MemRange[Int64, self.k_stage * 2] # type: ignore
            load_v_mbar_ptr: cute.struct.MemRange[Int64, self.v_stage * 2] # type: ignore
            # Masking
            mma_decay_mbar_ptr: cute.struct.MemRange[Int64, self.decay_stage * 2] # type: ignore
            mma_mask_mbar_ptr: cute.struct.MemRange[Int64, self.mma_mask_stage * 2] # type: ignore
            o_acc_mbar_ptr: cute.struct.MemRange[Int64, self.epi_stage * 2] # type: ignore
            # TMEM
            tmem_dealloc_mbar_ptr: cute.struct.MemRange[Int64, 1] # type: ignore
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Smem tensors
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, cute.cosize(o_smem_layout_staged)], # type: ignore
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(q_smem_layout_staged)], # type: ignore
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(k_smem_layout_staged)], # type: ignore
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(v_smem_layout_staged)], # type: ignore
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage        

        self.grid = self._compute_grid(
            o_shape=cute.shape(o),
            chunk_size=self.chunk_size,
        )

        self.kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            kv_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_o,
            tma_tensor_o,
            decay,
            q_smem_layout_staged,
            k_smem_layout_staged,
            kt_smem_layout_staged,
            v_smem_layout_staged,
            o_smem_layout_staged,
            p_tmem_layout_staged,
            self.chunk_size,
        ).launch(
            grid=self.grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        kv_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_dkl: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        mO_qdl: cute.Tensor,
        decay: cute.Pointer,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        kt_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        o_smem_layout_staged: cute.ComposedLayout,
        p_tmem_layout_staged: cute.ComposedLayout,
        chunk_size: int,
    ):
        """Kernel for linear attention.

        Args:
            qk_tiled_mma (cute.TiledMma): qk tiled mma
            kv_tiled_mma (cute.TiledMma): kv tiled mma
            pv_tiled_mma (cute.TiledMma): pv tiled mma
            tma_atom_q (cute.CopyAtom): _description_
            mQ_qdl (cute.Tensor): _description_
            tma_atom_k (cute.CopyAtom): _description_
            mK_kdl (cute.Tensor): _description_
            tma_atom_v (cute.CopyAtom): _description_
            mV_vdl (cute.Tensor): _description_
            tma_atom_o (cute.CopyAtom): _description_
            mO_odl (cute.Tensor): _description_
            decay (cute.Pointer): _description_
            q_smem_layout_staged (cute.ComposedLayout): _description_
            k_smem_layout_staged (cute.ComposedLayout): _description_
            v_smem_layout_staged (cute.ComposedLayout): _description_
            o_smem_layout_staged (cute.ComposedLayout): _description_
            chunk_size (int): _description_
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        # Prefetch TMA descriptors
        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o)

        # Allocate shared memory
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.q_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_q_bytes,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
        ).make_participants()
        load_k_producer, load_k_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.k_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_kv_bytes,
            barrier_storage=storage.load_k_mbar_ptr.data_ptr(),
        ).make_participants()
        load_v_producer, load_v_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.v_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_kv_bytes,
            barrier_storage=storage.load_v_mbar_ptr.data_ptr(),
        ).make_participants()
        mma_s0_producer, mma_s0_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_mask_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.decay_warp_ids)
            ),
            barrier_storage=storage.mma_mask_mbar_ptr.data_ptr(),
        ).make_participants()
        decay_producer, decay_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.decay_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.decay_warp_ids)
            ),
            barrier_storage=storage.mma_decay_mbar_ptr.data_ptr(),
        ).make_participants()

        # TMEM
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.data_ptr()        

        # decay barrier init
        if warp_idx == self.empty_warp_id:
            cute.arch.mbarrier_init(
                tmem_dealloc_mbar_ptr,
                self.threads_per_warp
                * len((
                    *self.decay_warp_ids,
                ))
            )
        cute.arch.mbarrier_init_fence()

        # Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, STAGE_Q)
        # sQ: ((64,16),1,(4,2),2):((64,1),0,(16,4096),8192)>
        sQ = storage.sQ.get_tensor(
            q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner
        )
        # (MMA, MMA_K, MMA_D, STAGE_K)
        # sK: tensor<ptr<bf16, smem, align<1024>, S<3,4,3>> o
        # ((64,16),1,(4,2),2):((64,1),0,(16,4096),8192)>
        sK = storage.sK.get_tensor(
            k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE_V)
        # sV: tensor<ptr<bf16, smem, align<1024>, S<3,4,3>> o
        # (((64,2),16),1,4,2):(((1,4096),64),0,1024,8192)>
        sV = storage.sV.get_tensor(
            v_smem_layout_staged.outer, swizzle=v_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE_O)
        # sO: tensor<ptr<bf16, smem, align<1024>, S<3,4,3>> o
        # ((8,16),(64,2),(1,2)):((64,512),(1,8192),(0,16384))>
        sO = storage.sO.get_tensor(
            o_smem_layout_staged.outer, swizzle=o_smem_layout_staged.inner
        )
        # 1-CTA
        qk_thr_mma = qk_tiled_mma.get_slice(0)
        pv_thr_mma = pv_tiled_mma.get_slice(0)
        kv_thr_mma = kv_tiled_mma.get_slice(0)
        # MMA SMEM descriptors
        tSrQ = qk_thr_mma.make_fragment_A(sQ)
        tSrK = qk_thr_mma.make_fragment_B(sK)
        tSrV = pv_thr_mma.make_fragment_B(sV)

        # (MMA, 1, 1)
        # qk_acc_shape: ((64,64),1,1)
        qk_acc_shape = qk_thr_mma.partition_shape_C(
            (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
        )
        # qk fragment C
        # tStS: tensor<ptr<f32, tmem, align<1>> o (((16,4),64),1,1):(((65536,2097152),1),0,0)>
        # Create tmem for S = QK
        tStS = qk_thr_mma.make_fragment_C(qk_acc_shape)

        # pv_acc_shape: ((128,128),1,1)
        pv_acc_shape = pv_thr_mma.partition_shape_C(
            (self.pv_mma_tiler[0], self.pv_mma_tiler[1])
        )
        # tOtO: tensor<ptr<f32, tmem, align<1>> o ((128,128),1,1):((65536,1),0,0)>
        # Create tmem for O = PV
        tOtO = pv_thr_mma.make_fragment_C(pv_acc_shape)

        tStS0 = cute.make_tensor(tStS.iterator + self.tmem_s0_offset, tStS.layout)
        tStS1 = cute.make_tensor(tStS.iterator + self.tmem_s1_offset, tStS.layout)
        tOtO0 = cute.make_tensor(tOtO.iterator + self.tmem_o0_offset, tOtO.layout)
        tOtO1 = cute.make_tensor(tOtO.iterator + self.tmem_o1_offset, tOtO.layout)

        tP = cute.make_tensor(tStS.iterator, p_tmem_layout_staged.outer)
        # tPfragA = pv_thr_mma.make_fragment_A(tP)

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=0,
            num_threads=self.threads_per_cta,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            # TODO
            allocator_warp_id=self.load_warp_id,
        )
        tmem.allocate(self.tmem_total_cols)

        # barrier before retrieve tmem ptr from smem
        tmem.wait_for_alloc()

        # Since we store acc to tmem
        tmem_ptr_base = tmem.retrieve_ptr(self.qk_acc_dtype)

        print(f"sQ: {cute.pretty_str(sQ)}")
        print(f"sK: {cute.pretty_str(sK)}")
        print(f"sV: {cute.pretty_str(sV)}")
        print(f"sO: {cute.pretty_str(sO)}")
        print(f"qk_thr_mma: {cute.pretty_str(qk_thr_mma)}")
        print(f"pv_thr_mma: {cute.pretty_str(pv_thr_mma)}")
        print(f"kv_thr_mma: {cute.pretty_str(kv_thr_mma)}")
        print(f"tSrQ: {cute.pretty_str(tSrQ)}")
        print(f"tSrK: {cute.pretty_str(tSrK)}")
        print(f"tSrV: {cute.pretty_str(tSrV)}")
        print(f"tStS: {cute.pretty_str(tStS)}")
        print(f"tOtO: {cute.pretty_str(tOtO)}")
        print(f"qk_acc_shape: {cute.pretty_str(qk_acc_shape)}")
        print(f"pv_acc_shape: {cute.pretty_str(pv_acc_shape)}")
        print(f"tStS0: {tStS0}")
        print(f"tStS1: {tStS1}")
        print(f"tOtO0: {tOtO0}")
        print(f"tOtO1: {tOtO1}")
        print(f"tP: {cute.pretty_str(tP)}")
        # print(f"tPfragA: {cute.pretty_str(tPfragA)}")

        # tOrP = pv_thr_mma.make_fragment_A(tP)[None, None, None, 0]
        # tOrP0 = cute.make_tensor(
        #    tOrP.iterator
        #    + self.qk_acc_dtype.width // self.q_dtype.width * self.tmem_p0_offset,
        #    tOrP.layout,
        # )
        # tOrP1 = cute.make_tensor(
        #     tOrP.iterator
        #     + self.qk_acc_dtype.width // self.q_dtype.width * self.tmem_p1_offset,
        #     tOrP.layout,
        # )
        # self.cta_sync_barrier.arrive_and_wait()

        self.num_regs_other = 32

        (_, hidx, bidx) = cute.arch.block_idx()

        # ///////////////////////////////////////////////////////////////////////////////
        # LOAD WARP
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

            # Load warp handles data loading for the entire chunk
            # Uses async copy or TMA to bring Q, K, V from global to shared memory

            # ((ATOM_V, REST_V), INPUT_STAGE)
            # ((ATOM_V, REST_V), 1, 1, C, EH, B)
            tQsQ, tQgQ_pre_slice = self.tma_partition_for_mma_a_operand(
                tma_atom_q,
                mQ_qdl,
                sQ,
                tiled_mma_intra2,
                cluster_layout_vmnk,
                mma_tile_coord_v,
                block_in_cluster_coord_vmnk,
            )

            mQ_qdl_ = mQ_qdl
            mK_kdl_ = mK_kdl
            mV_dkl_ = mV_dkl

            # mQ_qdl_: (0,0,0,0) o (4096,128,(64,2)):(1@1,1@0,(1@2,1@3))
            # mK_kdl_: (0,0,0,0) o (4096,128,(64,2)):(1@1,1@0,(1@2,1@3))
            # mV_dkl_: (0,0,0,0) o (128,4096,(64,2)):(1@0,1@1,(1@2,1@3))
            # gQ_qdl: (0,0,0,0) o (64,128,64,1,(64,2)):(1@1,1@0,64@1,128@0,(1@2,1@3))
            # gK_kdl: (0,0,0,0) o (64,128,64,1,(64,2)):(1@1,1@0,64@1,128@0,(1@2,1@3))
            # gV_dkl: (0,0,0,0) o (128,64,1,64,(64,2)):(1@0,1@1,128@0,64@1,(1@2,1@3))
            # tSgQ_qdl: (0,0,0,0) o ((64,16),1,8,64,1,(64,2)):((1@1,1@0),0,16@0,64@1,128@0,(1@2,1@3))
            # tSgK_kdl: (0,0,0,0) o ((64,16),1,8,64,1,(64,2)):((1@1,1@0),0,16@0,64@1,128@0,(1@2,1@3))
            # tSgV_dkl: (0,0,0,0) o ((128,16),1,4,1,64,(64,2)):((1@0,1@1),0,16@1,128@0,64@1,(1@2,1@3))
            # tQgQ: (0,0,0,0) o (((64,64),2),64):(((1@0,1@1),64@0),64@1)
            # tQgQ_qdl: (0,0,0,0) o (((64,64),2),64,1,(64,2)):(((1@0,1@1),64@0),64@1,128@0,(1@2,1@3))
            # tKgK: (0,0,0,0) o (((64,64),2),64):(((1@0,1@1),64@0),64@1)
            # tKgK_kdl: (0,0,0,0) o (((64,64),2),64,1,(64,2)):(((1@0,1@1),64@0),64@1,128@0,(1@2,1@3))
            # tVgV: (0,0,0,0) o (((64,64),2),64):(((1@0,1@1),64@0),64@1)
            # tVgV_dkl: (0,0,0,0) o (((64,64),2),1,64,(64,2)):(((1@0,1@1),64@0),128@0,64@1,(1@2,1@3))

            # Local tile partition global tensors
            # mQ_qdl_: (0,0,0,0) o (4096,128,(64,2)):(1@1,1@0,(1@2,1@3))
            # gQ_qdl: (0,0,0,0) o (64,128,64,1,(64,2)):(1@1,1@0,64@1,128@0,(1@2,1@3))
            # (bM, bK, loopM, loopK, loopL)
            gQ_qdl = cute.flat_divide(
                mQ_qdl_, cute.select(self.qk_mma_tiler, mode=[0, 2])
            )
            # tSgQ_qdl: (0,0,0,0) o ((64,16),1,8,64,1,(64,2)):((1@1,1@0),0,16@0,64@1,128@0,(1@2,1@3))
            # (MMA, MMA_Q, MMA_D, loopM, loopK, loopL)
            tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)

            # Tiles the GMEM and SMEM tensors for the provided TMA Copy Atom.
            #
            # sQ: ((64,16),1,(4,2),2):((64,1),0,(16,4096),8192)>
            # smem_tensor: (((64,16),1,(4,2)),2):(((64,1),0,(16,4096)),8192) =
            # tSgQ_qdl: ((64,16),1,8,64,1,(64,2)):((1@1,1@0),0,16@0,64@1,128@0,(1@2,1@3))
            # gmem_tensor: (((64,16),1,8),64,1,(64,2)):(((1@1,1@0),0,16@0),64@1,128@0,(1@2,1@3))
            # tma_atom_q: TV Layout Src: (1,4096):(0,1)
            # tQsQ: o ((4096,2),2):((1,4096),8192)
            # tQgQ_qdl: (((64,64),2),64,1,(64,2)):(((1@0,1@1),64@0),64@1,128@0,(1@2,1@3))
            # tQgQ: (((64,64),2),64):(((1@0,1@1),64@0),64@1)
            tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
                atom=tma_atom_q,
                cta_coord=0, # no multicast
                cta_layout=cute.make_layout(1),
                smem_tensor=cute.group_modes(sQ, 0, 3), # single stage smem tensor
                gmem_tensor=cute.group_modes(tSgQ_qdl, 0, 3),
            )
            tQgQ = tQgQ_qdl[None, None, 0, bidx]

            gK_kdl = cute.flat_divide(
                mK_kdl_, cute.select(self.qk_mma_tiler, mode=[1, 2])
            )
            tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
            tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
                tma_atom_k,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sK, 0, 3),
                cute.group_modes(tSgK_kdl, 0, 3),
            )
            tKgK = tKgK_kdl[None, None, 0, bidx]

            gV_dkl = cute.flat_divide(
                mV_dkl_, cute.select(self.pv_mma_tiler, mode=[1, 2])
            )
            tSgV_dkl = pv_thr_mma.partition_B(gV_dkl)
            tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
                tma_atom_v,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sV, 0, 3),
                cute.group_modes(tSgV_dkl, 0, 3),
            )
            tVgV = tVgV_dkl[None, 0, None, bidx]

            if bidx == 0 and hidx == 0 and tidx == self.load_warp_id * self.threads_per_warp:
                cute.printf("tidx: {}", tidx)
                cute.printf("mQ_qdl_: {}", mQ_qdl_)
                cute.printf("mK_kdl_: {}", mK_kdl_)
                cute.printf("mV_dkl_: {}", mV_dkl_)
                cute.printf("gQ_qdl: {}", gQ_qdl)
                cute.printf("gK_kdl: {}", gK_kdl)
                cute.printf("gV_dkl: {}", gV_dkl)

                cute.printf("tSgQ_qdl: {}", tSgQ_qdl)
                cute.printf("tSgK_kdl: {}", tSgK_kdl)
                cute.printf("tSgV_dkl: {}", tSgV_dkl)

                cute.printf("tQsQ: {}", tQsQ)
                cute.printf("tKsK: {}", tKsK)
                cute.printf("tVsV: {}", tVsV)

                cute.printf("tQgQ: {}", tQgQ)
                cute.printf("tQgQ_qdl: {}", tQgQ_qdl)
                cute.printf("tKgK: {}", tKgK)
                cute.printf("tKgK_kdl: {}", tKgK_kdl)
                cute.printf("tVgV: {}", tVgV)
                cute.printf("tVgV_dkl: {}", tVgV_dkl)

                cute.printf("smem_tensor: {}", cute.group_modes(sQ, 0, 3),)
                cute.printf("gmem_tensor: {}", cute.group_modes(tSgQ_qdl, 0, 3))

            assert self.S % self.chunk_size == 0, "Sequence length must be divisible by chunk size"

            # TODO: Add for loop to load each Qi, Ki, Vi, i for chunk idx
            for chunk_start in cutlass.range(0, self.S, self.chunk_size, unroll=1):
                # Qi
                idx = chunk_start // self.chunk_size
                q_handle = load_q_producer.acquire_and_advance()
                cute.copy(
                    atom=tma_atom_q,
                    src=tQgQ[None, idx], # source
                    dst=tQsQ[None, q_handle.index], # which stage
                    tma_bar_ptr=q_handle.barrier,
                )

                # Ki
                k_handle = load_k_producer.acquire_and_advance()
                cute.copy(
                    atom=tma_atom_k,
                    src=tKgK[None, idx],
                    dst=tKsK[None, k_handle.index],
                    tma_bar_ptr=k_handle.barrier,
                )

                # Vi
                v_handle = load_v_producer.acquire_and_advance()
                cute.copy(
                    atom=tma_atom_v,
                    src=tVgV[None, idx],
                    dst=tVsV[None, v_handle.index],
                    tma_bar_ptr=v_handle.barrier,
                )

        # ///////////////////////////////////////////////////////////////////////////////
        # COMPUTE WARPS
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            # Compute warp group processes chunkwise linear attention
            # Each warp computes a portion of the sequence in chunks

            # Allocate TMEM
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
            self.tmem_alloc_barrier.arrive_and_wait()

            for chunk_start in cutlass.range(0, self.S, self.chunk_size, unroll=1):
                # Process chunk from chunk_start to chunk_start + chunk_size
                idx = chunk_start // self.chunk_size

                # 1. Wait for Qi.
                q_handle = load_q_consumer.wait_and_advance()
                tSrQ0 = tSrQ[None, None, None, q_handle.index]
                # 2. Wait for Ki.
                k_handle = load_k_consumer.wait_and_advance()
                tSrK0 = tSrK[None, None, None, k_handle.index]
                # 3. Acquire empty S0 buffer
                s0_handle = mma_s0_producer.acquire_and_advance()
                # 4. GEMM
                num_kphases = cute.size(tSrQ0, mode=[2])
                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                    kphase_coord = (None, None, kphase_idx)
                    qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                    cute.gemm(
                        atom=qk_tiled_mma,
                        d=tStS0,
                        a=tSrQ0[kphase_coord],
                        b=tSrK0[kphase_coord],
                        c=tStS0,
                    )
                # 5. Release S0.
                s0_handle.commit()
                # End of GEMM (Qi, Ki) -> S0i

        # ///////////////////////////////////////////////////////////////////////////////
        # CORRECTION WARPS
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx in self.decay_warp_ids:

            # Correction warps perform numerical stability fixes and refinements
            #
            # Tasks:
            # 1. Handle epsilon addition for division stability
            #    denom' = denom + epsilon  to avoid division by zero
            # 2. Apply log-sum-exp tricks if needed for numerical stability
            # 3. Post-process output (clipping, normalization)
            # 4. Load/store synchronization

            # Wait for compute warps to finish intermediate computations
            pass

        # ///////////////////////////////////////////////////////////////////////////////
        # EMPTY WARP - Synchronization
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.empty_warp_id:
            pass    

        return

    def tma_partition_for_mma_b_operand(
        self,
        tma_atom_x,
        tma_tensor_x,
        smem_x,
        tile_shape,
        tiled_mma,
    ):
        _, hidx, bidx = cute.arch.block_idx()
        # Local_tile partition global tensors
        # (MMATile_N, MMATile_K, TILES_N, TILES_K)
        # mK_kdl_: (0,0,0,0) o (4096,128,(64,2)):(1@1,1@0,(1@2,1@3))
        gX = cute.local_tile(
            tma_tensor_x,
            tile_shape[1:],
            (None, None, (hidx, bidx)),
        )
        # Partition global tensor with regard to TiledMMA
        thr_mma = tiled_mma.get_slice(0)
        # tCgX: (MMA, MMA_N, MMA_K, TILES_N, TILES_K)
        tCgX = thr_mma.partition_B(gX)

        # ((ATOM_V, REST_V), INPUT_STAGE)
        # ((ATOM_V, REST_V), TILES_N, TILES_K)
        tXsX, tXgX_pre_slice = cute.nvgpu.cpasync.tma_partition(
            tma_atom_x,
            0, # no multicast
            cute.make_layout(1),
            cute.group_modes(smem_x, 0, 3),
            cute.group_modes(tCgX, 0, 3),
        )
        return tXsX, tXgX_pre_slice

    def tma_partition_for_mma_a_operand(
        self,
        tma_atom_x,
        tma_tensor_x,
        smem_x,
        tile_shape,
        tiled_mma,
    ):
        _, hidx, bidx = cute.arch.block_idx()
        # Local_tile partition global tensors
        # (MMATile_M, MMATile_K, TILES_M, TILES_K)
        # mK_kdl_: (0,0,0,0) o (4096,128,(64,2)):(1@1,1@0,(1@2,1@3))
        gX = cute.local_tile(
            tma_tensor_x,
            cute.slice_(tile_shape, (None, 0, None)), # MK
            (None, None, (hidx, bidx)),
        )
        # Partition global tensor with regard to TiledMMA
        thr_mma = tiled_mma.get_slice(0)
        # tCgX: (MMA, MMA_N, MMA_K, TILES_N, TILES_K)
        tCgX = thr_mma.partition_A(gX)

        # ((ATOM_V, REST_V), INPUT_STAGE)
        # ((ATOM_V, REST_V), TILES_N, TILES_K)
        tXsX, tXgX_pre_slice = cute.nvgpu.cpasync.tma_partition(
            tma_atom_x,
            0, # no multicast
            cute.make_layout(1),
            cute.group_modes(smem_x, 0, 3),
            cute.group_modes(tCgX, 0, 3),
        )
        return tXsX, tXgX_pre_slice

def make_thread_cooperative_group(size: int):
    """Helper to create thread cooperative groups for pipeline synchronization."""
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, size)


def main():
    """
    Example usage of LinearAttentionChunkwise with CuTe DSL
    """
    parser = argparse.ArgumentParser(
        description="Chunkwise Linear Attention with Headwise Decay"
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--num_heads", type=int, default=64, help="Number of heads")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--chunk_size", type=int, default=64, help="Chunk size")
    parser.add_argument("--decay", type=float, default=0.95, help="Decay factor")
    parser.add_argument(
        "--io_dtype", type=cutlass.dtype, default=cutlass.BFloat16,
        help="Input/output data type"
    )
    parser.add_argument(
        "--acc_dtype", type=cutlass.dtype, default=cutlass.Float32,
        help="Accumulation data type"
    )
    parser.add_argument(
        "--warmup_iterations", type=int, default=0, help="Warmup iterations"
    )
    parser.add_argument(
        "--iterations", type=int, default=1, help="Benchmark iterations"
    )
    
    args = parser.parse_args()
    
    print("Running Chunkwise Linear Attention with CuTe DSL:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Number of heads: {args.num_heads}")
    print(f"  Head dimension: {args.head_dim}")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  Decay factor: {args.decay}")
    print(f"  IO dtype: {args.io_dtype}")
    print(f"  Accumulation dtype: {args.acc_dtype}")
    print(f"  Warmup iterations: {args.warmup_iterations}")
    print(f"  Benchmark iterations: {args.iterations}")
    
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return
    
    # Create inputs
    B, S, H, D = args.batch_size, args.seq_len, args.num_heads, args.head_dim
    
    # Input tensors in format [B, S, H, D]
    Q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    
    # Per-head decay coefficients [H]
    decay = torch.full((H,), args.decay, device="cuda", dtype=torch.float32)
    
    # Convert to dlpack for CuTe
    q_cute = from_dlpack(Q)
    k_cute = from_dlpack(K)
    v_cute = from_dlpack(V)
    decay_cute = from_dlpack(decay)
    
    o_cute = from_dlpack(torch.zeros_like(Q))
    
    # Create kernel instance
    attn_kernel = LinearAttentionChunkwise(
        chunk_size=args.chunk_size,
        qk_acc_dtype=args.acc_dtype,
        kv_acc_dtype=args.acc_dtype,
        io_dtype=args.io_dtype,
    )

    # Get default stream
    stream = cutlass_torch.default_stream()

    start_time = time.time()
    compiled = cute.compile(
        attn_kernel,
        q_cute.iterator,
        k_cute.iterator,
        v_cute.iterator,
        o_cute.iterator,
        decay_cute.iterator,
        (Int32(B), Int32(S), Int32(H), Int32(D)),
        stream,
    )
    compilation_time = time.time() - start_time
    print(f"Compilation time: {compilation_time:.4f} seconds")

    # Warmup
    for _ in range(args.warmup_iterations):
        compiled(
            q_cute.iterator,
            k_cute.iterator,
            v_cute.iterator,
            o_cute.iterator,
            decay_cute.iterator,
            (Int32(B), Int32(S), Int32(H), Int32(D)),
            stream,
        )
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(args.iterations):
        compiled(
            q_cute.iterator,
            k_cute.iterator,
            v_cute.iterator,
            o_cute.iterator,
            decay_cute.iterator,
            (Int32(B), Int32(S), Int32(H), Int32(D)),
            stream,
        )
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    print(f"\nExecution time: {elapsed*1000/args.iterations:.2f} ms (average over {args.iterations} iterations)")
    print(f"Throughput: {(B*S*H*D*args.iterations) / (elapsed*1e9):.2f} GB/s")
    print("\nPASS")


if __name__ == "__main__":
    main()
