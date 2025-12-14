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
        # TODO: READ from input
        C, D = (64, 128)
        # (C, C, D)
        self.qk_mma_tiler = (C, C, D)  # (M, N, K)
        # (C, D, C)
        self.pv_mma_tiler = (C, D, C)  # (M, N, K)
        # (D, D, C)
        self.kv_mma_tiler = (D, D, C)  # (M, N, K)

        # one-cta cluster shape
        self.cluster_shape_mnk = (1, 1, 1)
        # For masking & decay.
        self.cuda_warp_ids = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.load_warp_id = 5
        self.epilogue_warp_id = 6
        self.empty_warp_id = 7

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.cuda_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.epilogue_warp_id,
            )
        )

        self.tmem_dealloc_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_cta,
        )

        self.buffer_align_bytes = 1024

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
        print(f"tCtAccQK_fake={tCtAccQK_fake}, num_qk_acc_cols={num_qk_acc_cols}")

        acc_shape_pv = tiled_mma_pv.partition_shape_C(tile_shape_mnk_pv[:2])
        tCtAccPV_fake = tiled_mma_pv.make_fragment_C(
            cute.append(acc_shape_pv, acc_stages)
        )
        num_pv_acc_cols = tcgen05.find_tmem_tensor_col_offset(tCtAccPV_fake)
        print(f"tCtAccPV_fake={tCtAccPV_fake}, num_pv_acc_cols={num_pv_acc_cols}")

        # No stage for linear state.
        acc_shape_kv = tiled_mma_kv.partition_shape_C(tile_shape_mnk_kv[:2])
        tCtAccKV_fake = tiled_mma_kv.make_fragment_C(
            cute.append(acc_shape_kv, 1)
        )
        num_kv_acc_cols = tcgen05.find_tmem_tensor_col_offset(tCtAccKV_fake)
        print(f"tCtAccKV_fake={tCtAccKV_fake}, num_kv_acc_cols={num_kv_acc_cols}")

        # For P.
        tCtP_fake = tiled_mma_qk.make_fragment_C(
            cute.append(acc_shape_qk, acc_stages)
        )
        num_p_cols = tcgen05.find_tmem_tensor_col_offset(tCtP_fake)
        print(f"tCtP_fake={tCtP_fake}, num_p_cols={num_p_cols}")

        num_qk_acc_cols_offset = 0
        num_pv_acc_cols_offset = num_qk_acc_cols_offset + num_qk_acc_cols
        num_kv_acc_cols_offset = num_pv_acc_cols_offset + num_pv_acc_cols
        num_p_cols_offset      = num_kv_acc_cols_offset + num_kv_acc_cols

        num_tmem_cols_total_tmp = num_p_cols_offset + num_p_cols
        # Turn num_tmem_cols_total to the nearest power of 2
        num_tmem_cols_total = 1
        while num_tmem_cols_total < num_tmem_cols_total_tmp:
            num_tmem_cols_total *= 2
        assert num_tmem_cols_total <= SM100_TMEM_CAPACITY_COLS

        print(f"num_qk_acc_cols_offset: {num_qk_acc_cols_offset}")
        print(f"num_pv_acc_cols_offset: {num_pv_acc_cols_offset}")
        print(f"num_kv_acc_cols_offset: {num_kv_acc_cols_offset}")
        print(f"num_p_cols_offset: {num_p_cols_offset}")
        print(f"num_tmem_cols_total: {num_tmem_cols_total}")

        # assert tile_shape_mnk_qk[1] * acc_stages == num_qk_acc_cols
        # assert tile_shape_mnk_pv[1] * acc_stages == num_pv_acc_cols
        # assert tile_shape_mnk_kv[1] * 1 == num_kv_acc_cols
        # assert tile_shape_mnk_qk[1] * acc_stages == num_p_cols

        return (
            num_qk_acc_cols_offset,
            num_pv_acc_cols_offset,
            num_kv_acc_cols_offset,
            num_p_cols_offset,
            num_tmem_cols_total,
        )

    def _setup_attributes(self):
        """Set up configurations and parameters for the linear attention kernel."""
        self.q_stage = 2
        self.k_stage = 2
        self.v_stage = 2
        self.o_stage = 2
        self.epi_stage = 2
        self.acc_stage = 2

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
        # kt
        # kt_layout = cute.make_layout(
        #     (S, D, (H,B)),
        #     stride=(D*H, 1, (D, D*H*S)),
        # )
        # kt = cute.make_tensor(k_iter, kt_layout)
        kt_layout = cute.make_layout(
            (D, S, (H,B)),
            stride=(1, D*H, (D, D*H*S)),
        )
        kt = cute.make_tensor(k_iter, kt_layout)
        # v
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
        self.kt_major_mode = utils.LayoutEnum.from_tensor(kt).mma_major_mode()
        self.v_major_mode = utils.LayoutEnum.from_tensor(v).mma_major_mode()
        self.o_layout = utils.LayoutEnum.from_tensor(o)

        if cutlass.const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of q is not supported")
        if cutlass.const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of k is not supported")
        # if cutlass.const_expr(self.kt_major_mode != tcgen05.OperandMajorMode.MN):
        #     raise RuntimeError("The layout of kt is not supported")
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
            self.k_dtype,
            self.kt_major_mode,
            self.v_major_mode,
            self.kv_acc_dtype,
            self.cta_group,
            self.kv_mma_tiler[:2],
        )
        p_major_mode = tcgen05.OperandMajorMode.K
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
            self.tmem_p_cols_offset,
            self.tmem_total_cols,
        ) = self._plan_tmem_offsets(
            qk_tiled_mma,
            self.qk_mma_tiler,
            pv_tiled_mma,
            self.pv_mma_tiler,
            kv_tiled_mma,
            self.kv_mma_tiler,
            # Try double buffer
            self.acc_stage,
        )

        cluster_layout_vmnk = cute.tiled_divide(
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
            pv_tiled_mma,
            self.pv_mma_tiler,
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
            cluster_layout_vmnk.shape,
        )

        # TMA load for K
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            k,
            k_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            cluster_layout_vmnk.shape,
        )
        kt_smem_layout = cute.select(kt_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_kt, tma_tensor_kt = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            kt,
            kt_smem_layout,
            self.kv_mma_tiler,
            kv_tiled_mma,
            cluster_layout_vmnk.shape,
        )
        # TMA load for V
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            v,
            v_smem_layout,
            self.pv_mma_tiler,
            pv_tiled_mma,
            cluster_layout_vmnk.shape,
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
        v_copy_size = cute.size_in_bytes(self.v_dtype, v_smem_layout)
        kt_copy_size = cute.size_in_bytes(self.k_dtype, kt_smem_layout)
        self.tma_copy_q_bytes = q_copy_size
        self.tma_copy_k_bytes = k_copy_size        
        self.tma_copy_v_bytes = v_copy_size        
        self.tma_copy_kt_bytes = kt_copy_size        

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
        print(f"cluster_layout_vmnk: {cute.pretty_str(cluster_layout_vmnk)}")
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
        print(f"tma_atom_kt: {cute.pretty_str(tma_atom_kt)}")
        print(f"tma_atom_v: {cute.pretty_str(tma_atom_v)}")
        print(f"tma_atom_o: {cute.pretty_str(tma_atom_o)}")
        print(f"tma_tensor_q: {cute.pretty_str(tma_tensor_q)}")
        print(f"tma_tensor_k: {cute.pretty_str(tma_tensor_k)}")
        print(f"tma_tensor_kt: {cute.pretty_str(tma_tensor_kt)}")
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
            load_kt_mbar_ptr: cute.struct.MemRange[Int64, self.k_stage * 2] # type: ignore
            load_v_mbar_ptr: cute.struct.MemRange[Int64, self.v_stage * 2] # type: ignore
            # Masking
            s_mbar_ptr: cute.struct.MemRange[Int64, self.acc_stage * 2] # type: ignore
            p_mbar_ptr: cute.struct.MemRange[Int64, self.acc_stage * 2] # type: ignore
            o_intra_mbar_ptr: cute.struct.MemRange[Int64, self.acc_stage * 2] # type: ignore
            o_inter_mbar_ptr: cute.struct.MemRange[Int64, self.acc_stage * 2] # type: ignore
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
            # TODO: reuse smem K
            sKT: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(kt_smem_layout_staged)], # type: ignore
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(v_smem_layout_staged)], # type: ignore
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage        
        print(f"size of storage: {SharedStorage.__sizeof__()}")

        self.grid = self._compute_grid(
            o_shape=cute.shape(o),
            chunk_size=self.chunk_size,
        )

        self.kernel(
            qk_tiled_mma,
            kv_tiled_mma,
            pv_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_kt,
            tma_tensor_kt,
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
        tma_atom_kt: cute.CopyAtom,
        mKT_kdl: cute.Tensor,
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
            tma_atom_kt (cute.CopyAtom): _description_
            mKT_kdl (cute.Tensor): _description_
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
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_kt)
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
            tx_count=self.tma_copy_k_bytes,
            barrier_storage=storage.load_k_mbar_ptr.data_ptr(),
        ).make_participants()
        load_kt_producer, load_kt_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.k_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_kt_bytes,
            barrier_storage=storage.load_kt_mbar_ptr.data_ptr(),
        ).make_participants()
        load_v_producer, load_v_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.v_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_v_bytes,
            barrier_storage=storage.load_v_mbar_ptr.data_ptr(),
        ).make_participants()
        mma_s0_producer, mma_s0_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.cuda_warp_ids)
            ),
            barrier_storage=storage.s_mbar_ptr.data_ptr(),
        ).make_participants()
        p_producer, p_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_stage,
            producer_group=make_thread_cooperative_group(len(self.cuda_warp_ids)),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len([self.mma_warp_id])
            ),
            barrier_storage=storage.p_mbar_ptr.data_ptr(),
        ).make_participants()
        o_intra_producer, o_intra_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.cuda_warp_ids)
            ),
            barrier_storage=storage.o_intra_mbar_ptr.data_ptr(),
        ).make_participants()
        o_inter_producer, o_inter_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.cuda_warp_ids)
            ),
            barrier_storage=storage.o_inter_mbar_ptr.data_ptr(),
        ).make_participants()

        # TMEM
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.load_warp_id,
        )
        tmem.allocate(self.tmem_total_cols)

        # Barrier before retrieve tensor memory ptr from shared memory
        tmem.wait_for_alloc()

        # Retrieve tmem ptr
        tmem_ptr_base = tmem.retrieve_ptr(self.qk_acc_dtype)

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
        sKT = storage.sKT.get_tensor(
            kt_smem_layout_staged.outer, swizzle=kt_smem_layout_staged.inner
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

        print(f"sQ: {cute.pretty_str(sQ)}")
        print(f"sK: {cute.pretty_str(sK)}")
        print(f"sKT: {cute.pretty_str(sKT)}")
        print(f"sV: {cute.pretty_str(sV)}")
        print(f"sO: {cute.pretty_str(sO)}")

        self.num_regs_other = 24
        self.num_regs_uniform_warps = 24
        self.num_regs_pre_inter_warps = 168
        self.num_regs_pre_intra_warps = 208
        self.num_regs_epilogue_warps = 112
        self.num_regs_mma = 64
        self.num_regs_cuda = 192

        (_, hidx, bidx) = cute.arch.block_idx()
        B, S, H, D, C = self.B, self.S, self.H, self.D, self.chunk_size

        # ///////////////////////////////////////////////////////////////////////////////
        # LOAD WARP
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_alloc(self.num_regs_cuda)

            # ((ATOM_V, REST_V), INPUT_STAGE)
            # ((ATOM_V, REST_V), TILES_N, TILES_K)
            tQsQ, tQgQ = self.tma_partition_for_mma_operand(
                tma_atom_q,
                mQ_qdl,
                sQ,
                self.qk_mma_tiler,
                qk_tiled_mma,
                operand_mode="A",
                debug_name="Q",
            )

            tKsK, tKgK = self.tma_partition_for_mma_operand(
                tma_atom_k,
                mK_kdl,
                sK,
                self.qk_mma_tiler,
                qk_tiled_mma,
                operand_mode="B",
                debug_name="K",
            )

            tKsKT, tKgKT = self.tma_partition_for_mma_operand(
                tma_atom_k,
                mKT_kdl,
                sKT,
                self.kv_mma_tiler,
                kv_tiled_mma,
                operand_mode="A",
                debug_name="KT",
            )

            tVsV, tVgV = self.tma_partition_for_mma_operand(
                tma_atom_v,
                mV_dkl,
                sV,
                self.pv_mma_tiler,
                pv_tiled_mma,
                operand_mode="B",
                debug_name="V",
            )

            if bidx == 0 and hidx == 0 and tidx == self.load_warp_id * self.threads_per_warp:
                cute.printf("tidx: {}", tidx)
                cute.printf("mQ_qdl: {}", mQ_qdl)
                cute.printf("mK_kdl: {}", mK_kdl)
                cute.printf("mV_dkl: {}", mV_dkl)

                cute.printf("tQsQ: {}", tQsQ)
                cute.printf("tQgQ: {}", tQgQ)
                cute.printf("tKsK: {}", tKsK)
                cute.printf("tKgK: {}", tKgK)
                cute.printf("tKsKT: {}", tKsKT)
                cute.printf("tKgKT: {}", tKgKT)
                cute.printf("tVsV: {}", tVsV)
                cute.printf("tVgV: {}", tVgV)


            # TODO: Add for loop to load each Qi, Ki, Vi, i for chunk idx
            for chunk_start in cutlass.range(0, 4096, C, unroll=0):
                # Chunk iterate over TILES_M, TILES_K is 1 in our case since max D is 128
                idx = chunk_start // C
                # print(f"S={S}, C={C}, S/C={S/C}, chunk_start={chunk_start}")

                # Qi
                # SRC: ((ATOM_V, REST_V), TILES_M, TILES_K)
                # DST: ((ATOM_V, REST_V), INPUT_STAGE)
                q_handle = load_q_producer.acquire_and_advance()
                if tidx == warp_idx * 32 and hidx == 0 and bidx == 0:
                    cute.printf("q producer: idx={}", idx)
                cute.copy(
                    atom=tma_atom_q,
                    src=tQgQ[None, idx, 0], # source
                    dst=tQsQ[None, q_handle.index], # which stage
                    tma_bar_ptr=q_handle.barrier,
                )


                # Ki
                # SRC: ((ATOM_V, REST_V), TILES_N, TILES_K)
                # DST: ((ATOM_V, REST_V), INPUT_STAGE)
                k_handle = load_k_producer.acquire_and_advance()
                if tidx == warp_idx * 32 and hidx == 0 and bidx == 0:
                    cute.printf("k producer: idx={}", idx)
                cute.copy(
                    atom=tma_atom_k,
                    src=tKgKT[None, idx, 0],
                    dst=tKsKT[None, k_handle.index],
                    tma_bar_ptr=k_handle.barrier,
                )

                # KTi
                # SRC: ((ATOM_V, REST_V), TILES_N, TILES_K)
                # DST: ((ATOM_V, REST_V), INPUT_STAGE)
                # TODO: check layout
                kt_handle = load_kt_producer.acquire_and_advance()
                if tidx == warp_idx * 32 and hidx == 0 and bidx == 0:
                    cute.printf("kt producer: idx={}", idx)
                cute.copy(
                    atom=tma_atom_kt,
                    src=tKgKT[None, idx, 0],
                    dst=tKsKT[None, kt_handle.index],
                    tma_bar_ptr=kt_handle.barrier,
                )

                # Vi
                # SRC: ((ATOM_V, REST_V), TILES_M, TILES_K)
                # DST: ((ATOM_V, REST_V), INPUT_STAGE)
                v_handle = load_v_producer.acquire_and_advance()
                if tidx == warp_idx * 32 and hidx == 0 and bidx == 0:
                    cute.printf("v producer: idx={}", idx)
                cute.copy(
                    atom=tma_atom_v,
                    src=tVgV[None, idx, 0],
                    dst=tVsV[None, v_handle.index],
                    tma_bar_ptr=v_handle.barrier,
                )

        # ///////////////////////////////////////////////////////////////////////////////
        # COMPUTE WARPS
        # ///////////////////////////////////////////////////////////////////////////////
        elif warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_mma)

            # Make fragments/tmem for QK MMA.
            # (MMA, MMA_M, MMA_K, INPUT_STAGE)
            # (MMA, MMA_N, MMA_K, INPUT_STAGE)
            # (MMA, MMA_M, MMA_N, ACC_STAGE)
            tCrQ, tCrK, tCtAccQK = self.mma_partition_ss(
                qk_tiled_mma,
                self.qk_mma_tiler,
                sQ,
                sK,
                tmem_ptr_base + self.tmem_qk_cols_offset,
                self.acc_stage,
            )

            # Make fragments/tmem for KV MMA.
            # (MMA, MMA_M, MMA_K, INPUT_STAGE)
            # (MMA, MMA_N, MMA_K, INPUT_STAGE)
            # (MMA, MMA_M, MMA_N, ACC_STAGE)
            print(f"kv_tiled_mma: {kv_tiled_mma}")
            print(f"kv_mma_tiler: {kv_tiled_mma}")
            print(f"sKT: {sKT}")
            print(f"sV: {sV}")
            tCrKT, tCrV, tCtAccKV = self.mma_partition_ss(
                kv_tiled_mma,
                self.kv_mma_tiler,
                sKT,
                sV,
                tmem_ptr_base + self.tmem_kv_cols_offset,
                1, # no stage for state accum
            )

            # Make fragments/tmem for PV MMA.
            # (MMA, MMA_M, MMA_K, INPUT_STAGE)
            # (MMA, MMA_N, MMA_K, INPUT_STAGE)
            # (MMA, MMA_M, MMA_N, ACC_STAGE)
            tCrP, tCrV2, tCtAccPV = self.mma_partition_ts(
                pv_tiled_mma,
                self.pv_mma_tiler,
                p_tmem_layout_staged,
                sV,
                tmem_ptr_base + self.tmem_p_cols_offset,
                tmem_ptr_base + self.tmem_pv_cols_offset,
                self.acc_stage,
            )

            for chunk_start in cutlass.range(0, 4096, C, unroll=0):
                # Process chunk from chunk_start to chunk_start + chunk_size
                idx = chunk_start // C

                # 1. Wait for Qi.
                q_handle = load_q_consumer.wait_and_advance()
                if tidx == warp_idx * 32 and hidx == 0 and bidx == 0:
                    cute.printf("q consumer: idx={}", idx)
                # 2. Wait for Ki.
                k_handle = load_k_consumer.wait_and_advance()
                if tidx == warp_idx * 32 and hidx == 0 and bidx == 0:
                    cute.printf("k consumer: idx={}", idx)
                # 3. Acquire empty S0 buffer
                s0_handle = mma_s0_producer.acquire_and_advance()
                if tidx == warp_idx * 32 and hidx == 0 and bidx == 0:
                    cute.printf("s0 producer: idx={}", idx)
                # 4. GEMM
                qk_tiled_mma = self.exec_mma(
                    tiled_mma=qk_tiled_mma,
                    tCtAcc=tCtAccQK,
                    tCrA=tCrQ,
                    tCrB=tCrK,
                    a_stage_idx=q_handle.index,
                    b_stage_idx=k_handle.index,
                    acc_stage_idx=s0_handle.index,
                )
                if tidx == warp_idx * 32 and hidx == 0 and bidx == 0:
                    cute.printf("after qk mma: idx={}", idx)
                # 5. Release S0.
                q_handle.release()
                k_handle.release()
                s0_handle.commit()
                # End of GEMM (Qi, Ki) -> S0i

                # Wait for V
                v_handle = load_v_consumer.wait_and_advance()
                if tidx == warp_idx * 32 and hidx == 0 and bidx == 0:
                    cute.printf("v consumer: idx={}", idx)

                # Produce new_state
                # Wait for Ki^T
                if tidx == warp_idx * 32 and hidx == 0 and bidx == 0:
                    cute.printf("-- begin wait for kt consumer: idx={}", idx)
                kt_handle = load_kt_consumer.wait_and_advance()
                if tidx == warp_idx * 32 and hidx == 0 and bidx == 0:
                    cute.printf("kt consumer: idx={}", idx)
                kv_tiled_mma = self.exec_mma(
                    tiled_mma=kv_tiled_mma,
                    tCtAcc=tCtAccKV,
                    tCrA=tCrKT,
                    tCrB=tCrV,
                    a_stage_idx=kt_handle.index,
                    b_stage_idx=v_handle.index,
                    acc_stage_idx=0,
                    always_acc=True, # always accumulate states
                )
                
                kt_handle.release()
                if tidx == warp_idx * 32 and hidx == 0 and bidx == 0:
                    cute.printf("after kv mma: idx={}", idx)

                # Acquire empty state buffer.
                # TODO: Produce o_inter = gemm(q, state)

                # Produce o_intra = gemm(p, v)
                p_handle = p_consumer.wait_and_advance()
                if tidx == warp_idx * 32 and hidx == 0 and bidx == 0:
                    cute.printf("p consumer: idx={}", idx)
                o_intra_handle = o_intra_producer.acquire_and_advance()
                if tidx == warp_idx * 32 and hidx == 0 and bidx == 0:
                    cute.printf("o_intra producer: idx={}", idx)

                pv_tiled_mma = self.exec_mma(
                    tiled_mma=pv_tiled_mma,
                    tCtAcc=tCtAccPV,
                    tCrA=tCrP,
                    tCrB=tCrV2,
                    a_stage_idx=p_handle.index,
                    b_stage_idx=v_handle.index,
                    acc_stage_idx=o_intra_handle.index,
                )

                if tidx == warp_idx * 32 and hidx == 0 and bidx == 0:
                    cute.printf("after pv mma: idx={}", idx)

                p_handle.release()
                o_intra_handle.commit()

                # Release V here
                v_handle.release()

                # TODO: 



        # ///////////////////////////////////////////////////////////////////////////////
        # CUDA CORE WARPS
        # ///////////////////////////////////////////////////////////////////////////////
        elif warp_idx in self.cuda_warp_ids:
            cute.arch.warpgroup_reg_alloc(self.num_regs_cuda)

            for chunk_start in cutlass.range(0, 4096, C, unroll=0):

                idx = chunk_start // C

                if tidx == warp_idx * 32 and hidx == 0 and bidx == 0 and warp_idx == self.cuda_warp_ids[0]:
                    cute.printf("-- begin cuda_warp: idx={}", idx)


                # Wait for qk
                s0_handle = mma_s0_consumer.wait_and_advance()
                if tidx == warp_idx * 32 and hidx == 0 and bidx == 0 and warp_idx == self.cuda_warp_ids[0]:
                    cute.printf("s0 consumer: idx={}", idx)
                # Write P=Mask(QK) back to TMEM
                p_handle = p_producer.acquire_and_advance()
                if tidx == warp_idx * 32 and hidx == 0 and bidx == 0 and warp_idx == self.cuda_warp_ids[0]:
                    cute.printf("p producer: idx={}", idx)
                # TODO: impl p

                s0_handle.release()
                p_handle.commit()

                # O INTRA
                # TODO: o = o_intra + o_inter
                o_intra_handle = o_intra_consumer.wait_and_advance()
                o_intra_handle.release()
                if tidx == warp_idx * 32 and hidx == 0 and bidx == 0 and warp_idx == self.cuda_warp_ids[0]:
                    cute.printf("o_intra consumer: idx={}", idx)

            
        # ///////////////////////////////////////////////////////////////////////////////
        # EMPTY WARP - Synchronization
        # ///////////////////////////////////////////////////////////////////////////////
        elif warp_idx == self.empty_warp_id:
            pass

        else:
            pass

        # Release tensor memory allocation lock
        tmem.relinquish_alloc_permit()
        # Sync before deallocating tmem
        self.tmem_dealloc_sync_barrier.arrive_and_wait()
        # Dealloc tmem buffer
        tmem.free(tmem_ptr_base)

        return

    @cute.jit
    def mma_partition_ss(
        self,
        tiled_mma,
        tile_shape_mnk,
        smem_a,
        smem_b,
        tmem_acc_ptr,
        acc_stages,
    ):
        # (MMA, MMA_M, MMA_K, INPUT_STAGE)
        tCrA = tiled_mma.make_fragment_A(smem_a)
        # (MMA, MMA_N, MMA_K, INPUT_STAGE)
        tCrB = tiled_mma.make_fragment_B(smem_b)
        # (MMA, MMA_M, MMA_N, ACC_STAGE)
        tCtAcc = self.mma_partition_c(
            tiled_mma, tile_shape_mnk, tmem_acc_ptr, acc_stages
        )
        return tCrA, tCrB, tCtAcc

    @cute.jit
    def mma_partition_ts(
        self,
        tiled_mma,
        tile_shape_mnk,
        a_tmem_layout,
        smem_b,
        tmem_a_ptr,
        tmem_acc_ptr,
        acc_stages,
    ):
        # (MMA, MMA_M, MMA_K, INTERNAL_STAGE)
        tCrA = self.mma_partition_a_tmem(tiled_mma, a_tmem_layout, tmem_a_ptr)
        # (MMA, MMA_N, MMA_K, INPUT_STAGE)
        tCrB = tiled_mma.make_fragment_B(smem_b)
        # (MMA, MMA_M, MMA_N, INTERNAL_STAGE)
        tCtAcc = self.mma_partition_c(
            tiled_mma, tile_shape_mnk, tmem_acc_ptr, acc_stages
        )
        return tCrA, tCrB, tCtAcc

    @cute.jit
    def mma_partition_a_tmem(self, tiled_mma, a_tmem_layout, tmem_a_ptr):
        tCrA_fake = tiled_mma.make_fragment_A(a_tmem_layout.outer.shape)
        tCrA = cute.make_tensor(
            cute.recast_ptr(
                tmem_a_ptr,
                dtype=tCrA_fake.element_type,
            ),
            tCrA_fake.layout,
        )
        return tCrA

    @cute.jit
    def mma_partition_c(self, tiled_mma, tile_shape_mnk, tmem_acc_ptr, acc_stages):
        acc_shape = tiled_mma.partition_shape_C(tile_shape_mnk[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, acc_stages))
        # (MMA, MMA_M, MMA_N, INTERNAL_STAGE)
        tCtAcc = cute.make_tensor(tmem_acc_ptr, tCtAcc_fake.layout)
        return tCtAcc

    @cute.jit
    def exec_mma(
        self,
        tiled_mma,
        tCtAcc,
        tCrA,
        tCrB,
        a_stage_idx,
        b_stage_idx,
        acc_stage_idx,
        always_acc=False,
    ):
        for kphase_idx in cutlass.range(cute.size(tCrB, mode=[2]), unroll_full=True):
            # set accu = 1
            tiled_mma.set(
                tcgen05.Field.ACCUMULATE,
                cutlass.Boolean(kphase_idx != 0 or always_acc),
            )
            cute.gemm(
                tiled_mma,
                tCtAcc[None, None, None, acc_stage_idx],
                tCrA[None, None, kphase_idx, a_stage_idx],
                tCrB[None, None, kphase_idx, b_stage_idx],
                tCtAcc[None, None, None, acc_stage_idx],
            )
        return tiled_mma

    @cute.jit
    def tma_partition_for_mma_operand(
        self,
        tma_atom_x,
        tma_tensor_x,
        smem_x,
        tile_shape,
        tiled_mma,
        operand_mode,
        debug_name=None,
    ):
        _, hidx, bidx = cute.arch.block_idx()
        # Local_tile partition global tensors
        # x: (0,0,0,0) o (M,K,(H,B)):(1@1,1@0,(1@2,1@3))
        # (MMATile_M, MMATile_K, TILES_M, TILES_K, (H, B))
        operand_mode = operand_mode.upper()
        coord = None
        if cutlass.const_expr(operand_mode == "B"):
            coord = (0, None, None) 
        elif cutlass.const_expr(operand_mode == "C"):
            coord = (None, None, 0)
        elif cutlass.const_expr(operand_mode == 'A'):
            coord = (None, 0, None) 
        else:
            raise RuntimeError(f"unknown operand mode: {operand_mode}")
            
        gX = cute.local_tile(
            tma_tensor_x,
            cute.slice_(tile_shape, coord), # MK, (64, 128)
            (None, None, (hidx, bidx)),
        )
        # Partition global tensor with regard to TiledMMA
        thr_mma = tiled_mma.get_slice(0)
        # tCgX: (MMA, MMA_M, MMA_K, TILES_M, TILES_K)
        if cutlass.const_expr(operand_mode == 'A'):
            tCgX = thr_mma.partition_A(gX)
        elif cutlass.const_expr(operand_mode == 'B'):
            tCgX = thr_mma.partition_B(gX)
        elif cutlass.const_expr(operand_mode == 'C'):
            tCgX = thr_mma.partition_C(gX)
        else:
            raise RuntimeError("unknown operand mode")

        # ((ATOM_V, REST_V), INPUT_STAGE)
        # ((ATOM_V, REST_V), TILES_N, TILES_K)
        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            tma_atom_x,
            0, # no multicast
            cute.make_layout(1),
            cute.group_modes(smem_x, 0, 3),
            cute.group_modes(tCgX, 0, 3),
        )
        print("===========================: {}", debug_name)
        print(f"gX: {gX}")
        print(f"tma_tensor_x: {tma_tensor_x}")
        print(f"thr_mma: {thr_mma}")
        print(f"tCgX: {tCgX}")
        print(f"tXsX: {tXsX}")
        print(f"tXgX: {tXgX}")
        return tXsX, tXgX

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
