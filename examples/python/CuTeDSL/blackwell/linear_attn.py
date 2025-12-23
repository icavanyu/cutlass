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
from typing import Type, Tuple, List, Union

import torch
import torch.nn.functional as F
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.cute.testing as testing
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32, Int64, Float32

PRINT_DEBUG=False

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
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        io_dtype: Type[cutlass.Numeric] = cutlass.BFloat16,
    ):
        self.chunk_size = chunk_size
        self.qk_acc_dtype = qk_acc_dtype
        self.kv_acc_dtype = kv_acc_dtype
        self.pv_acc_dtype = kv_acc_dtype
        self.acc_dtype = acc_dtype
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
        # (D, C, C)
        self.vp_mma_tiler = (D, C, C)  # (M, N, K)
        # (D, D, C)
        self.kv_mma_tiler = (D, D, C)  # (M, N, K)
        # (D, C, D)
        # State as operand A since it's in TMEM
        # Q now as operand B
        self.sq_mma_tiler = (D, C, D)  # (M, N, K)

        # one-cta cluster shape
        self.cluster_shape_mnk = (1, 1, 1)
        # For masking & decay.
        self.cuda_warp_ids = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.load_warp_id = 5
        # self.epilogue_warp_id = 6
        # self.empty_warp_id = 7

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.cuda_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                # self.epilogue_warp_id,
            )
        )

        self.tmem_dealloc_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_cta,
        )

        self.buffer_align_bytes = 1024
        self.use_tma_store = False

    @staticmethod
    def _plan_tmem_offsets(
        tiled_mma_qk,
        tile_shape_mnk_qk,
        tiled_mma_pv,
        tile_shape_mnk_pv,
        tiled_mma_kv,
        tile_shape_mnk_kv,
        tiled_mma_sq,
        tile_shape_mnk_sq,
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
        num_kv16_acc_cols = num_kv_acc_cols // 2  # BF16 has half columns
        print(f"tCtAccKV_fake={tCtAccKV_fake}, num_kv_acc_cols={num_kv_acc_cols}, num_kv16_acc_cols={num_kv16_acc_cols}")

        # No stage for linear state.
        acc_shape_sq = tiled_mma_sq.partition_shape_C(tile_shape_mnk_sq[:2])
        tCtAccSQ_fake = tiled_mma_sq.make_fragment_C(
            cute.append(acc_shape_sq, 1)
        )
        num_sq_acc_cols = tcgen05.find_tmem_tensor_col_offset(tCtAccSQ_fake)
        print(f"tCtAccSQ_fake={tCtAccSQ_fake}, num_sq_acc_cols={num_sq_acc_cols}")

        # For P. P has half the columns of QK accumulator since its BF16.
        num_p_cols = num_qk_acc_cols // 2
        print(f"num_p_cols={num_p_cols}")

        num_qk_acc_cols_offset = 0
        num_pv_acc_cols_offset = num_qk_acc_cols_offset + num_qk_acc_cols
        num_kv_acc_cols_offset = num_pv_acc_cols_offset + num_pv_acc_cols
        num_kv16_acc_cols_offset = num_kv_acc_cols_offset + num_kv_acc_cols
        num_qs_acc_cols_offset = num_kv16_acc_cols_offset + num_kv16_acc_cols

        # Reuse TMEM-QK for P
        num_p_cols_offset = num_qk_acc_cols_offset

        num_tmem_cols_total_tmp = num_qs_acc_cols_offset + num_sq_acc_cols
        # Turn num_tmem_cols_total to the nearest power of 2
        num_tmem_cols_total = 1
        while num_tmem_cols_total < num_tmem_cols_total_tmp:
            num_tmem_cols_total *= 2
        assert num_tmem_cols_total <= SM100_TMEM_CAPACITY_COLS

        print(f"num_qk_acc_cols_offset: {num_qk_acc_cols_offset}")
        print(f"num_pv_acc_cols_offset: {num_pv_acc_cols_offset}")
        print(f"num_kv_acc_cols_offset: {num_kv_acc_cols_offset}")
        print(f"num_kv16_acc_cols_offset: {num_kv16_acc_cols_offset}")
        print(f"num_p_cols_offset: {num_p_cols_offset}")
        print(f"num_tmem_cols_total: {num_tmem_cols_total}")

        return (
            num_qk_acc_cols_offset,
            num_pv_acc_cols_offset,
            num_kv_acc_cols_offset,
            num_kv16_acc_cols_offset,
            num_qs_acc_cols_offset,
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
        # problem_size: Tuple[int, int, int, int],  # (B, S, H, D)
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

        # Setup attributes
        self._setup_attributes()

        # TODO: try two-cta
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
        # v
        # v_layout = cute.make_layout(
        #     (D, S, (H,B)),
        #     stride=(1, D*H, (D, D*H*S)),
        # )
        v_layout = cute.make_layout(
            (S, D, (H,B)),
            stride=(D*H, 1, (D, D*H*S)),
        )
        v = cute.make_tensor(v_iter, v_layout)

        # (S, D, (H,B))
        # o_layout = cute.make_layout(
        #     (S, D, (H,B)),
        #     stride=(D*H, 1, (D, D*H*S)),
        # )
        # (D, S, (H,B))
        o_layout = cute.make_layout(
             (D, S, (H,B)),
             stride=(1, D*H, (D, D*H*S)),
        )
        o = cute.make_tensor(o_iter, o_layout)

        # TODO: output final state
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
        self.k_major_mode_kv = tcgen05.OperandMajorMode.MN  # For V^T*K, S dimension coalesced
        # TMEM register output results as (D, C)
        self.o_layout = utils.LayoutEnum.from_tensor(o)

        if cutlass.const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of q is not supported")
        if cutlass.const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of k is not supported")
        if cutlass.const_expr(self.o_layout != utils.LayoutEnum.COL_MAJOR):
            raise RuntimeError("The layout of o is not supported")
        if cutlass.const_expr(self.k_major_mode == self.k_major_mode_kv):
            raise RuntimeError("The layout of k & k^t should be different")

        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            self.cta_group,
            self.qk_mma_tiler[:2],
        )
        # V^T*K, majorness
        kv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.k_dtype,
            self.v_major_mode,
            self.k_major_mode_kv,
            self.kv_acc_dtype,
            self.cta_group,
            self.kv_mma_tiler[:2],
        )
        # State^T Q^T
        sq_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            # State is in TMEM, always K major, TODO
            tcgen05.OperandMajorMode.K,
            self.q_major_mode,
            self.qk_acc_dtype,
            self.cta_group,
            self.sq_mma_tiler[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        p_major_mode = tcgen05.OperandMajorMode.K
        vp_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            self.v_major_mode,
            p_major_mode,
            self.pv_acc_dtype,
            self.cta_group,
            self.vp_mma_tiler[:2],
        )

        (
            self.tmem_qk_cols_offset,
            self.tmem_pv_cols_offset,
            self.tmem_kv_cols_offset,
            self.tmem_kv16_cols_offset,
            self.tmem_sq_cols_offset,
            self.tmem_total_cols,
        ) = self._plan_tmem_offsets(
            qk_tiled_mma,
            self.qk_mma_tiler,
            vp_tiled_mma,
            self.vp_mma_tiler,
            kv_tiled_mma,
            self.kv_mma_tiler,
            sq_tiled_mma,
            self.sq_mma_tiler,
            # Try double buffer
            self.acc_stage,
        )

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )

        # Output shape, (D, C)
        self.epi_tile = (self.vp_mma_tiler[0], self.vp_mma_tiler[1]) # pv
        self.qk_epi_tile = (self.qk_mma_tiler[0], self.qk_mma_tiler[1]) # qk

        # Q&K^T
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
        # V^T*K
        v_smem_layout_staged = sm100_utils.make_smem_layout_a(
            vp_tiled_mma,
            self.vp_mma_tiler,
            self.v_dtype,
            self.v_stage,
        )
        kv_k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            kv_tiled_mma,
            self.kv_mma_tiler,
            self.k_dtype,
            self.k_stage,
        )
        # V^T*P
        p_smem_layout_staged = sm100_utils.make_smem_layout_b(
            vp_tiled_mma,
            self.vp_mma_tiler,
            self.v_dtype,
            self.acc_stage,
        )
        state_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            sq_tiled_mma,
            self.sq_mma_tiler,
            self.q_dtype,
            num_stages=1,
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
        kv_k_smem_layout = cute.select(kv_k_smem_layout_staged, mode=[0, 1, 2])
        # tma_atom_kt, tma_tensor_kt = cute.nvgpu.make_tiled_tma_atom_A(
        #     tma_load_op,
        #     kt,
        #     kv_k_smem_layout,
        #     self.kv_mma_tiler,
        #     kv_tiled_mma,
        #     cluster_layout_vmnk.shape,
        # )
        # TMA load for V
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            v,
            v_smem_layout,
            self.vp_mma_tiler,
            vp_tiled_mma,
            cluster_layout_vmnk.shape,
        )
        # TMA store for O
        ## o_smem_layout = cute.select(o_smem_layout_staged, mode=[0, 1, 2])
        ## tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
        ##     tma_store_op,
        ##     o,
        ##     o_smem_layout,
        ##     self.epi_tile,
        ## )

        q_copy_size = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        k_copy_size = cute.size_in_bytes(self.k_dtype, k_smem_layout)
        v_copy_size = cute.size_in_bytes(self.v_dtype, v_smem_layout)
        self.tma_copy_q_bytes = q_copy_size
        self.tma_copy_k_bytes = k_copy_size        
        self.tma_copy_v_bytes = v_copy_size        

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
        print(f"vp_tiled_mma: {cute.pretty_str(vp_tiled_mma)}")
        print(f"sq_tiled_mma: {cute.pretty_str(sq_tiled_mma)}")
        print(f"cluster_layout_vmnk: {cute.pretty_str(cluster_layout_vmnk)}")
        print(f"epi_tile: {cute.pretty_str(self.epi_tile)}")
        print(f"q_smem_layout: {cute.pretty_str(q_smem_layout)}")
        print(f"k_smem_layout: {cute.pretty_str(k_smem_layout)}")
        print(f"v_smem_layout: {cute.pretty_str(v_smem_layout)}")
        ## print(f"o_smem_layout: {cute.pretty_str(o_smem_layout)}")
        print(f"q_smem_layout_staged: {cute.pretty_str(q_smem_layout_staged)}")
        print(f"k_smem_layout_staged: {cute.pretty_str(k_smem_layout_staged)}")
        print(f"kv_k_smem_layout_staged: {cute.pretty_str(kv_k_smem_layout_staged)}")
        print(f"v_smem_layout_staged: {cute.pretty_str(v_smem_layout_staged)}")
        print(f"o_smem_layout_staged: {cute.pretty_str(o_smem_layout_staged)}")
        print(f"p_smem_layout_staged: {cute.pretty_str(p_smem_layout_staged)}")
        print(f"tma_atom_q: {cute.pretty_str(tma_atom_q)}")
        print(f"tma_atom_k: {cute.pretty_str(tma_atom_k)}")
        print(f"tma_atom_v: {cute.pretty_str(tma_atom_v)}")
        ## print(f"tma_atom_o: {cute.pretty_str(tma_atom_o)}")
        print(f"tma_tensor_q: {cute.pretty_str(tma_tensor_q)}")
        print(f"tma_tensor_k: {cute.pretty_str(tma_tensor_k)}")
        print(f"tma_tensor_v: {cute.pretty_str(tma_tensor_v)}")
        ## print(f"tma_tensor_o: {cute.pretty_str(tma_tensor_o)}")
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
            # KV
            kv_mbar_ptr: cute.struct.MemRange[Int64, self.acc_stage * 2] # type: ignore
            kv16_mbar_ptr: cute.struct.MemRange[Int64, self.acc_stage * 2] # type: ignore
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
            # TODO: should be able to reuse smem K, plz check swizzle of k
            sKT: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(kv_k_smem_layout_staged)], # type: ignore
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(v_smem_layout_staged)], # type: ignore
                self.buffer_align_bytes,
            ]
            # Store QK
            sP: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(p_smem_layout_staged)], # type: ignore
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
            vp_tiled_mma,
            sq_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            o,
            decay,
            q_smem_layout_staged,
            k_smem_layout_staged,
            kv_k_smem_layout_staged,
            v_smem_layout_staged,
            o_smem_layout_staged,
            p_smem_layout_staged,
            state_tmem_layout_staged,
            problem_size,
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
        vp_tiled_mma: cute.TiledMma,
        sq_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        tma_tensor_q: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        tma_tensor_v: cute.Tensor,
        o: cute.Tensor,
        decay: cute.Pointer,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        kv_k_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        o_smem_layout_staged: cute.ComposedLayout,
        p_smem_layout_staged: cute.ComposedLayout,
        state_tmem_layout_staged: cute.ComposedLayout,
        problem_size: Tuple[Int32, Int32, Int32, Int32],  # (B, S, H, D)
    ):
        """Kernel for linear attention.

        Args:
            qk_tiled_mma (cute.TiledMma): qk tiled mma
            kv_tiled_mma (cute.TiledMma): kv tiled mma
            vp_tiled_mma (cute.TiledMma): pv tiled mma
            tma_atom_q (cute.CopyAtom): _description_
            tma_tensor_q (cute.Tensor): _description_
            tma_atom_k (cute.CopyAtom): _description_
            tma_tensor_k (cute.Tensor): _description_
            tma_atom_v (cute.CopyAtom): _description_
            mV_vdl (cute.Tensor): _description_
            o (cute.Tensor): _description_
            decay (cute.Pointer): _description_
            q_smem_layout_staged (cute.ComposedLayout): _description_
            k_smem_layout_staged (cute.ComposedLayout): _description_
            v_smem_layout_staged (cute.ComposedLayout): _description_
            o_smem_layout_staged (cute.ComposedLayout): _description_
            p_smem_layout_staged (cute.ComposedLayout): _description_
            chunk_size (int): _description_
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        # Prefetch TMA descriptors
        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            # cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o)

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
        # load_kt_producer, load_kt_consumer = pipeline.PipelineTmaUmma.create(
        #     num_stages=self.k_stage,
        #     producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
        #     consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
        #     tx_count=self.tma_copy_kt_bytes,
        #     barrier_storage=storage.load_kt_mbar_ptr.data_ptr(),
        # ).make_participants()
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
        # Notify cuda core to convert 32-bit accumulator to 16-bit
        kv_producer, kv_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id]),),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len(self.cuda_warp_ids)
            ),
            barrier_storage=storage.kv_mbar_ptr.data_ptr(),
        ).make_participants()
        # Notify mma warp that 16bit state is ready for mma as operand A
        kv16_producer, kv16_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(len(self.cuda_warp_ids),),
            consumer_group=make_thread_cooperative_group(
                self.threads_per_warp * len([self.mma_warp_id])
            ),
            barrier_storage=storage.kv16_mbar_ptr.data_ptr(),
        ).make_participants()
        p_producer, p_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_stage, # TODO: check p stages
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
        # TODO: Avoid duplicated loading of k even through L2 cache might make it fine.
        sK_kv = storage.sK.get_tensor(
            kv_k_smem_layout_staged.outer, swizzle=kv_k_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE_V)
        # sV: tensor<ptr<bf16, smem, align<1024>, S<3,4,3>> o
        # (((64,2),16),1,4,2):(((1,4096),64),0,1024,8192)>
        sV = storage.sV.get_tensor(
            v_smem_layout_staged.outer, swizzle=v_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sP = storage.sP.get_tensor(
            p_smem_layout_staged.outer, swizzle=p_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE_O)
        # sO: tensor<ptr<bf16, smem, align<1024>, S<3,4,3>> o
        # ((8,16),(64,2),(1,2)):((64,512),(1,8192),(0,16384))>
        sO = storage.sO.get_tensor(
            o_smem_layout_staged.outer, swizzle=o_smem_layout_staged.inner
        )

        print(f"sQ: {cute.pretty_str(sQ)}")
        print(f"sK: {cute.pretty_str(sK)}")
        print(f"sK_kv: {cute.pretty_str(sK_kv)}")
        print(f"sV: {cute.pretty_str(sV)}")
        print(f"sO: {cute.pretty_str(sO)}")
        print(f"sP: {cute.pretty_str(sP)}")

        self.num_regs_other = 24
        self.num_regs_uniform_warps = 24
        self.num_regs_pre_inter_warps = 168
        self.num_regs_pre_intra_warps = 208
        self.num_regs_epilogue_warps = 112
        self.num_regs_mma = 64
        self.num_regs_cuda = 192

        (_, hidx, bidx) = cute.arch.block_idx()
        B, S, H, D = problem_size
        C = self.chunk_size

        qk_thr_mma = qk_tiled_mma.get_slice(0)
        vp_thr_mma = vp_tiled_mma.get_slice(0)
        kv_thr_mma = kv_tiled_mma.get_slice(0)
        sq_thr_mma = sq_tiled_mma.get_slice(0)

        qk_acc_shape = qk_thr_mma.partition_shape_C(
            (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
        )
        tStS = qk_thr_mma.make_fragment_C(qk_acc_shape)

        # vp_acc_shape = vp_thr_mma.partition_shape_C(
        #     (self.vp_mma_tiler[0], self.vp_mma_tiler[1])
        # )
        # tOtO = vp_thr_mma.make_fragment_C(vp_acc_shape)
        tCgO = self.local_tile_partition_for_mma_operand(
            tensor_x=o,
            tile_shape=self.vp_mma_tiler,
            tiled_mma=vp_tiled_mma,
            operand_mode="C",
            debug_name="O",
        )

        kv_acc_shape = kv_thr_mma.partition_shape_C(
            (self.kv_mma_tiler[0], self.kv_mma_tiler[1])
        )
        tKVtKV = kv_thr_mma.make_fragment_C(kv_acc_shape)

        # No Stage
        # (MMA, MMA_M, MMA_N)
        tmem_s  = cute.make_tensor(tmem_ptr_base + self.tmem_qk_cols_offset, tStS.layout)
        tmem_kv  = cute.make_tensor(tmem_ptr_base + self.tmem_kv_cols_offset, tKVtKV.layout)
        tmem_kv16 = cute.make_tensor(
            cute.recast_ptr(tmem_ptr_base + self.tmem_kv16_cols_offset, dtype=self.io_dtype), tKVtKV.layout
        )

        #-------------------------------------------------------------
        # Make fragments for MMAs.

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
        tCrV, tCrK_kv, tCtAccKV = self.mma_partition_ss(
            kv_tiled_mma,
            self.kv_mma_tiler,
            sV,
            sK_kv,
            tmem_ptr_base + self.tmem_kv_cols_offset,
            1, # NOTE: no stage for state accum
        )

        tCrState = self.mma_partition_a_tmem(
            sq_tiled_mma,
            state_tmem_layout_staged,
            tmem_ptr_base + self.tmem_kv16_cols_offset,
        )
        # tCtState = sq_thr_mma.make_fragment_A(tCtAccKV16)[None, None, None, 0]

        # Make fragments/tmem for SQ MMA.
        # S comes from tCtAccKV
        # tCtState = sq_thr_mma.make_fragment_A(tCtAccKV16)[None, None, None, 0]
        # (MMA, MMA_N, MMA_K, INPUT_STAGE)
        tCrQ_sq = sq_tiled_mma.make_fragment_B(sQ)
        # (MMA, MMA_M, MMA_N, ACC_STAGE)
        tCtAccSQ = self.mma_partition_c(
            sq_tiled_mma,
            self.sq_mma_tiler,
            tmem_ptr_base + self.tmem_sq_cols_offset,
            1, # no stage for state accumulations & state dependent vars
        )

        # Make fragments/tmem for VP MMA.
        # (MMA, MMA_M, MMA_K, INPUT_STAGE)
        # (MMA, MMA_N, MMA_K, INPUT_STAGE)
        # (MMA, MMA_M, MMA_N, ACC_STAGE)
        tCrV_dup, tCrP, tCtAccPV = self.mma_partition_ss(
            vp_tiled_mma,
            self.vp_mma_tiler,
            sV,
            sP,
            tmem_ptr_base + self.tmem_pv_cols_offset,
            self.acc_stage,
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # LOAD WARP
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_alloc(self.num_regs_cuda)

            # ((ATOM_V, REST_V), INPUT_STAGE)
            # ((ATOM_V, REST_V), TILES_N, TILES_K)
            tQsQ, tQgQ = self.tma_partition_for_mma_operand(
                tma_atom_q,
                tma_tensor_q,
                sQ,
                self.qk_mma_tiler,
                qk_tiled_mma,
                operand_mode="A",
                debug_name="Q",
            )

            tKsK, tKgK = self.tma_partition_for_mma_operand(
                tma_atom_k,
                tma_tensor_k,
                sK,
                self.qk_mma_tiler,
                qk_tiled_mma,
                operand_mode="B",
                debug_name="K",
            )

            tVsV, tVgV = self.tma_partition_for_mma_operand(
                tma_atom_v,
                tma_tensor_v,
                sV,
                self.vp_mma_tiler,
                vp_tiled_mma,
                operand_mode="A",
                debug_name="V",
            )

            if cutlass.const_expr(PRINT_DEBUG):
                if bidx == 0 and hidx == 0 and tidx == self.load_warp_id * self.threads_per_warp:
                    cute.printf("tidx: {}", tidx)
                    cute.printf("tma_tensor_q: {}", tma_tensor_q)
                    cute.printf("tma_tensor_k: {}", tma_tensor_k)
                    cute.printf("tma_tensor_v: {}", tma_tensor_v)

                    cute.printf("tQsQ: {}", tQsQ)
                    cute.printf("tQgQ: {}", tQgQ)
                    cute.printf("tKsK: {}", tKsK)
                    cute.printf("tKgK: {}", tKgK)
                    cute.printf("tVsV: {}", tVsV)
                    cute.printf("tVgV: {}", tVgV)

            for chunk_start in cutlass.range(0, S, C, unroll=0):
                # Chunk iterate over TILES_M, TILES_K is 1 in our case since max D is 128
                idx = chunk_start // C

                # Qi
                # SRC: ((ATOM_V, REST_V), TILES_M, TILES_K)
                # DST: ((ATOM_V, REST_V), INPUT_STAGE)
                q_handle = load_q_producer.acquire_and_advance()
                if cutlass.const_expr(PRINT_DEBUG):
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
                if cutlass.const_expr(PRINT_DEBUG):
                    if tidx == warp_idx * 32 and hidx == 0 and bidx == 0:
                        cute.printf("k producer: idx={}", idx)
                cute.copy(
                    atom=tma_atom_k,
                    src=tKgK[None, idx, 0],
                    dst=tKsK[None, k_handle.index],
                    tma_bar_ptr=k_handle.barrier,
                )

                # Vi
                # SRC: ((ATOM_V, REST_V), TILES_M, TILES_K)
                # DST: ((ATOM_V, REST_V), INPUT_STAGE)
                v_handle = load_v_producer.acquire_and_advance()
                if cutlass.const_expr(PRINT_DEBUG):
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

            for chunk_start in cutlass.range(0, S, C, unroll=0):
                # Process chunk from chunk_start to chunk_start + chunk_size
                idx = chunk_start // C

                # Wait for Qi.
                q_handle = load_q_consumer.wait_and_advance()
                if cutlass.const_expr(PRINT_DEBUG):
                    if tidx == warp_idx * 32 and hidx == 0 and bidx == 0:
                        cute.printf("q consumer: idx={}", idx)

                if idx != 0:
                    kv16_handle = kv16_consumer.wait_and_advance()
                    o_inter_handle = o_inter_producer.acquire_and_advance()

                    # TODO: support initial state
                    # Compute SQ once Qi is ready.
                    sq_tiled_mma = self.exec_mma(
                        tiled_mma=sq_tiled_mma,
                        tCtAcc=tCtAccSQ,
                        tCrA=tCrState,
                        tCrB=tCrQ_sq,
                        a_stage_idx=0,
                        b_stage_idx=q_handle.index,
                        acc_stage_idx=0,
                    )
                    o_inter_handle.commit()
                    kv16_handle.release()
                        
                # Wait for Ki.
                k_handle = load_k_consumer.wait_and_advance()
                if cutlass.const_expr(PRINT_DEBUG):
                    if tidx == warp_idx * 32 and hidx == 0 and bidx == 0:
                        cute.printf("k consumer: idx={}", idx)
                # Acquire empty S0 buffer
                s0_handle = mma_s0_producer.acquire_and_advance()
                if cutlass.const_expr(PRINT_DEBUG):
                    if tidx == warp_idx * 32 and hidx == 0 and bidx == 0:
                        cute.printf("s0 producer: idx={}", idx)
                # GEMM
                qk_tiled_mma = self.exec_mma(
                    tiled_mma=qk_tiled_mma,
                    tCtAcc=tCtAccQK,
                    tCrA=tCrQ,
                    tCrB=tCrK,
                    a_stage_idx=q_handle.index,
                    b_stage_idx=k_handle.index,
                    acc_stage_idx=s0_handle.index,
                )
                if cutlass.const_expr(PRINT_DEBUG):
                    if tidx == warp_idx * 32 and hidx == 0 and bidx == 0:
                        cute.printf("after qk mma: idx={}", idx)
                # Release Q. 
                q_handle.release()
                # Commit S = QK.
                s0_handle.commit()
                # End of GEMM (Qi, Ki) -> S0i

                # Wait for PV, produce ointra
                v_handle = load_v_consumer.wait_and_advance()
                p_handle = p_consumer.wait_and_advance()
                o_intra_handle = o_intra_producer.acquire_and_advance()

                # both v and p are in smem
                vp_tiled_mma = self.exec_mma(
                    tiled_mma=vp_tiled_mma,
                    tCtAcc=tCtAccPV,
                    tCrA=tCrV,
                    tCrB=tCrP,
                    a_stage_idx=v_handle.index,
                    b_stage_idx=p_handle.index,
                    acc_stage_idx=o_intra_handle.index,
                )
                p_handle.release()
                o_intra_handle.commit()

                kv_handle = kv_producer.acquire_and_advance()
                # NOTE: Always ACC to avoid adding in cuda core.
                kv_tiled_mma = self.exec_mma(
                    tiled_mma=kv_tiled_mma,
                    tCtAcc=tCtAccKV,
                    tCrA=tCrV,
                    tCrB=tCrK_kv,
                    a_stage_idx=v_handle.index,
                    b_stage_idx=k_handle.index,
                    acc_stage_idx=0,
                    always_acc=True if idx != 0 else False, # always accumulate states
                )
                kv_handle.commit()
                # Release K V here
                k_handle.release()
                v_handle.release()

        # ///////////////////////////////////////////////////////////////////////////////
        # CUDA CORE WARPS
        # ///////////////////////////////////////////////////////////////////////////////
        elif warp_idx in self.cuda_warp_ids:
            cute.arch.warpgroup_reg_alloc(self.num_regs_cuda)

            #----------------------------------------------------------
            local_tidx = tidx % (self.threads_per_warp * len(self.cuda_warp_ids))

            debug = True if cutlass.const_expr(PRINT_DEBUG) and tidx == warp_idx * 32 and hidx == 0 and bidx == 0 and warp_idx == self.cuda_warp_ids[0] else False

            # constant mask tensor
            cM = cute.make_identity_tensor(self.qk_mma_tiler[:2])
            print(f"cM: {cM}")
            print(f"tmem_s: {tmem_s}")

            # With ACC_STAGE
            # O1
            (
                tiled_copy_t2r_pv,
                tTR_tAcc_base_pv,
                tTR_rAcc_pv,
            ) = self.epilog_tmem_copy_and_partition(
                tidx, tCtAccPV, tCgO, self.vp_mma_tiler, self.epi_tile, use_2cta_instrs=False
            )
            # Prepare copy from rmem to gmem
            # TODO: replace to TMA STORE
            # tCgX: (MMA, MMA_M, MMA_K, TILES_M, TILES_K)
            # tCgO: tensor<ptr<bf16, gmem> o ((128,64),1,1,?,?):((1,?),0,0,128,?{div=64})>
            print(f"tCgO: {tCgO}")
            (
                simt_atom_o, tTR_rO, tTR_gO_partitioned,
            ) = self.epilog_gmem_copy_and_partition(
                tidx=tidx,
                atom=tiled_copy_t2r_pv,
                gC_mnl=tCgO,
                epi_tile=self.epi_tile,
                sC=tCtAccPV,
                c_dtype=self.io_dtype,
                use_tma_store=False,
            )
            print(f"tTR_rO: {cute.pretty_str(tTR_rO)}")
            print(f"tTR_gO_partitioned: {cute.pretty_str(tTR_gO_partitioned)}")

            # O2, i.e. O_INTER
            # SQ: (128, 64), (D, C)
            (
                tiled_copy_t2r_sq,
                tTR_tAcc_base_sq,
                tTR_rAcc_sq,
            ) = self.epilog_tmem_copy_and_partition(
                tidx, tCtAccSQ, tCgO, self.sq_mma_tiler, self.epi_tile, use_2cta_instrs=False
            )
            print(f"tiled_copy_t2r_sq: {tiled_copy_t2r_sq}")
            print(f"tTR_tAcc_base_sq: {tTR_tAcc_base_sq}")
            print(f"tTR_rAcc_sq: {tTR_rAcc_sq}")

            # P = QK^T
            # A fixed shape CxC: 64x64, FP32
            # According to PTX, we need to use 16dp.
            # copy_atom_t2r_S = sm100_utils.get_tmem_load_op(
            #     self.qk_mma_tiler,
            #     utils.LayoutEnum.from_tensor(sP),
            #     self.io_dtype,
            #     # self.qk_acc_dtype,
            #     self.qk_acc_dtype,
            #     self.qk_mma_tiler[:2],
            #     use_2cta_instrs=False,
            # )
            copy_atom_t2r_S = cute.make_copy_atom(
                tcgen05.Ld16x256bOp(tcgen05.Repetition(8), tcgen05.Pack.NONE),
                self.qk_acc_dtype,
            )
            # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
            # TODO: figure out why direct tCtAccQK0 causes error
            tAccQK_epi = cute.flat_divide(
                tCtAccQK[((None, None), None, None, None)],
                self.qk_epi_tile,
            )
            # (EPI_TILE_M, EPI_TILE_N)
            tiled_t2r_S = tcgen05.make_tmem_copy(
                copy_atom_t2r_S, tAccQK_epi[(None, None, 0, 0, 0, 0, 0)]
            )
            # tiled_t2r_S = tcgen05.make_tmem_copy(copy_atom_t2r_S, tCtAccQK0)

            thr_t2r = tiled_t2r_S.get_slice(local_tidx)

            # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N)
            tTR_cS = thr_t2r.partition_D(qk_thr_mma.partition_C(cM))
            tTR_tS = thr_t2r.partition_S(tCtAccQK)
            tTR_rS = cute.make_rmem_tensor(
                tTR_cS.shape,
                self.qk_acc_dtype,
            )
            # create p
            tTR_rP = cute.make_rmem_tensor_like(
                src=tTR_rS,
                dtype=self.q_dtype,
            )

            # P has shape (C, C), where C = 64
            # 4 x 16dp x 16b x 64 = 16x256b x 4
            # 128 threads
            tiled_copy_r2s_P, tRS_rP, tRS_sP = self.smem_copy_and_partition(
                tiled_copy_t2r=tiled_t2r_S,
                qk_tiled_mma=qk_tiled_mma,
                tTR_rC=tTR_rP,
                tidx=tidx,
                sC=sP,
                c_layout=utils.LayoutEnum.from_tensor(sP),
                c_dtype=self.q_dtype,
                acc_dtype=self.qk_acc_dtype,
            )

            print(f"tCtAccQK: {tCtAccQK}")
            print(f"tTR_tS: {tTR_tS}")
            print(f"tTR_rS: {tTR_rS}")
            print(f"tTR_rP: {tTR_rP}")
            print(f"tRS_rP: {tRS_rP}")
            print(f"tRS_sP: {tRS_sP}")

            #-------------------------------------------------------

            # With ACC_STAGE
            # KV

            ### (
            ###     tiled_copy_t2r_kv,
            ###     tiled_copy_r2t_kv,
            ###     tTR_tKV,
            ###     tTR_rKV,
            ###     tRT_tKV16,
            ###     tRT_rKV16,
            ### ) = self.make_tmem_load_and_store_for_kv(
            ###     local_tidx, tCtAccKV, tCtAccKV16, self.kv_mma_tiler, self.kv_mma_tiler[:2], kv_thr_mma, use_2cta_instrs=False
            ### )

            tCtAccKV_slice = tCtAccKV[((None, None), 0, 0, None)]
            (
                tiled_copy_t2r_kv,
                tTR_tKV,
                tTR_rKV,
            ) = self.tmem_load_partition_kv(
                mma_tiler=self.kv_mma_tiler,
                tState=tCtAccKV_slice,
                local_tidx=local_tidx,
            )

            (
                tiled_copy_r2t_kv,
                tRT_tKV16,
                tRT_rKV16,
            ) = self.tmem_store_and_partition_kv(
                local_tidx, tCrState,
            )

            print(f"tiled_copy_t2r_kv: {tiled_copy_t2r_kv}")
            print(f"LOAD tTR_tKV: {tTR_tKV}")
            print(f"LOAD tTR_rKV: {tTR_rKV}")
            print(f"STORE tRT_tKV16: {tRT_tKV16}")
            print(f"STORE tRT_rKV16: {tRT_rKV16}")
            print(f"tiled_copy_r2t_kv16: {tiled_copy_r2t_kv}")

            #-------------------------------------------------------

            for chunk_start in cutlass.range(0, S, C, unroll=0):
                idx = chunk_start // C
                if debug:
                    cute.printf("-- begin cuda_warp: idx={}", idx)

                # Wait for S = QK^T
                s0_handle = mma_s0_consumer.wait_and_advance()
                if debug:
                    cute.printf("s0 consumer: idx={}", idx)

                # (MMA, MMA_M, MMA_N, ACC_STAGE)
                tTR_tSi = tTR_tS[None, None, None, None, s0_handle.index]
                # Load S from TMEM to RMEM
                cute.copy(tiled_t2r_S, tTR_tSi, tTR_rS)
                cute.arch.fence_view_async_tmem_load()

                # Apply mask and convert to BF16
                # TODO: check causal correctness
                self.apply_mask(tTR_rS, tTR_cS, tTR_rP, debug=False)

                # Write P to SMEM
                p_handle = p_producer.acquire_and_advance()
                if debug:
                        cute.printf("p producer: idx={}", idx)

                # Store P from RMEM to SMEM
                tRS_sPi = tRS_sP[(None, None, None, None, p_handle.index)]
                cute.copy(tiled_copy_r2s_P, tRS_rP, tRS_sPi)
                # Fence
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                s0_handle.release()
                p_handle.commit()

                # Convert KV to KV16
                if idx != 0:
                    kv_handle = kv_consumer.wait_and_advance()
                    tTR_tKVi = tTR_tKV[(None, None, None, kv_handle.index)] # kv stage == 1
                    cute.copy(tiled_copy_t2r_kv, tTR_tKVi, tTR_rKV)
                    cute.arch.fence_view_async_tmem_load()
                    kv_handle.release()

                    acc_vec = tTR_rKV.load() # NOTE: RETILE
                    # acc_vec = tiled_copy_r2t_kv.retile(tTR_rKV).load() # NOTE: RETILE
                    acc_vec = acc_vec.to(self.io_dtype)
                    tRT_rKV16.store(acc_vec)

                    kv16_handle = kv16_producer.acquire_and_advance()
                    tRT_tKV16i = tRT_tKV16[(None, None, None, None, kv_handle.index)] # kv stage == 1
                    cute.copy(tiled_copy_r2t_kv, tRT_rKV16, tRT_tKV16i)
                    kv16_handle.commit()

                # Wait for O_INTER
                if idx != 0:
                    o_inter_handle = o_inter_consumer.wait_and_advance()
                    tTR_tAcc_sq_i = tTR_tAcc_base_sq[(None, None, None, 0, 0, o_inter_handle.index)]
                    # Load O_INTER from TMEM to RMEM
                    cute.copy(tiled_copy_t2r_sq, tTR_tAcc_sq_i, tTR_rAcc_sq)
                    o_inter_handle.release()

                # Wait for O_INTRA
                o_intra_handle = o_intra_consumer.wait_and_advance()
                if debug:
                    cute.printf("o_intra consumer: idx={}", idx)
                
                # Load O_INTRA from TMEM to RMEM
                tTR_tAcc_pv_i = tTR_tAcc_base_pv[(None, None, None, 0, 0, o_intra_handle.index)]
                cute.copy(tiled_copy_t2r_pv, tTR_tAcc_pv_i, tTR_rAcc_pv)
                cute.arch.fence_view_async_tmem_load()
                o_intra_handle.release()

                # Perform addition and store to gmem
                acc_vec = tTR_rAcc_pv.load()
                acc_vec = acc_vec.to(self.io_dtype)
                if idx != 0:
                    acc_vec_inter = tTR_rAcc_sq.load()
                    acc_vec_inter = acc_vec_inter.to(self.io_dtype)
                    acc_vec = acc_vec + acc_vec_inter
                tTR_rO.store(acc_vec)

                # Store to gmem
                # tCgX: (MMA, MMA_M, MMA_N, TILES_M, TILES_N)
                # tCgO: tensor<ptr<bf16, gmem> o ((128,64),1,1,?,?):((1,?),0,0,128,?{div=64})>
                # tTR_rO: tensor<ptr<bf16, rmem, align<32>> o (((2,2,8),1),2,1):(((1,2,4),0),320#  )>
                # tTR_gO_partitioned: tensor<ptr<bf16, gmem> o (((2,2,8),1),2,1,1,1,?,?):(((?,8,?{div=8}),0),16,0,0,0,128,?{div=64                # })>
                # Output: (D,S), (_, _, _, EPI_M, EPI_N, TILES_M, TILES_N)
                tTR_gOi = tTR_gO_partitioned[(None, None, None, 0, 0, 0, idx)]
                cute.autovec_copy(tTR_rO, tTR_gOi)
                
        # ///////////////////////////////////////////////////////////////////////////////
        # EMPTY WARP - Synchronization
        # ///////////////////////////////////////////////////////////////////////////////
        else:
            pass

        # Release tensor memory allocation lock
        tmem.relinquish_alloc_permit()
        # Sync before deallocating tmem
        self.tmem_dealloc_sync_barrier.arrive_and_wait()
        # Dealloc tmem buffer
        tmem.free(tmem_ptr_base)

        return

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
        c_dtype: Type[cutlass.Numeric],
        use_tma_store: bool,
    ) -> tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """
        Partitions source and destination tensors for a global memory store.

        This method generates a tiled copy for storing results to global memory
        and partitions the source (register or shared memory) and destination
        (global memory) tensors accordingly. The behavior varies based on whether
        TMA store is enabled.

        :param tidx: The thread index in epilogue warp groups.
        :type tidx: cutlass.Int32
        :param atom: The copy atom to be used (TMA or universal).
        :type atom: cute.CopyAtom or cute.TiledCopy
        :param gC_mnl: The global tensor C.
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler.
        :type epi_tile: cute.Tile
        :param sC: The shared memory tensor C.
        :return: A tuple containing the appropriate copy atom and partitioned
                 source and destination tensors for the store operation.
        :rtype: tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        gC_epi = cute.flat_divide(
            # ((ATOM_V, REST_V), TILES_N, TILES_K)
            gC_mnl[((None, None), 0, 0, None, None)], epi_tile
        )
        print(f"gC_mnl: {cute.pretty_str(gC_mnl)}")
        print(f"gC_epi: {cute.pretty_str(gC_epi)}")
        if use_tma_store:
            tma_atom_c = atom
            sC_for_tma_partition = cute.group_modes(sC, 0, 2)
            gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
            # ((ATOM_V, REST_V), EPI_M, EPI_N)
            # ((ATOM_V, REST_V), EPI_M, EPI_N, RestM, RestN, RestL)
            bSG_sC, bSG_gC = cpasync.tma_partition(
                tma_atom_c,
                0,
                cute.make_layout(1),
                sC_for_tma_partition,
                gC_for_tma_partition,
            )
            return tma_atom_c, bSG_sC, bSG_gC
        else:
            tiled_copy_t2r = atom
            # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN)
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_gC = thr_copy_t2r.partition_D(gC_epi)
            # (T2R, T2R_M, T2R_N)
            tTR_rC = cute.make_rmem_tensor(
                tTR_gC[(None, None, None, 0, 0, 0, 0)].shape, c_dtype
            )
            simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), c_dtype)
            return simt_atom, tTR_rC, tTR_gC
            

    @cute.jit
    def smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        qk_tiled_mma: cute.TiledMma,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
        c_layout: cutlass.utils.LayoutEnum,
        c_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Partitions source and destination tensors for a shared memory store.

        This method generates a tiled copy for storing results to shared memory
        and partitions the source (register) and destination (shared memory)
        tensors accordingly.

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy.
        :param tTR_rC: The partitioned accumulator tensor.
        :param tidx: The thread index in epilogue warp groups.
        :param sC: The shared memory tensor to be copied and partitioned.
        :return: A tuple containing the tiled copy for the store operation and
                 the partitioned source and destination tensors.
        """

        copy_atom_r2s = sm100_utils.get_smem_store_op(
            c_layout, c_dtype, acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)

        print(f"------------ SMEM COPY AND PARTITION --------------")
        num_dp, num_bits, num_rep, pack = sm100_utils.get_tmem_copy_properties(tiled_copy_t2r)
        print(f"tmem copy properties: dp={num_dp}, bits={num_bits}, rep={num_rep}, pack={pack}")
        print(f"tiled_copy_t2r for P: {tiled_copy_t2r}")
        print(f"copy_atom_r2s for P: {copy_atom_r2s}")
        print(f"tiled_copy_r2s for P: {tiled_copy_r2s}")
        print(f"thr_copy_r2s for P: {thr_copy_r2s}")
        print(f"sC for P: {sC}")
        print(f"tRS_sC for P: {tRS_sC}")
        print(f"tRS_rC for P: {tRS_rC}")
        print(f"------------ SMEM COPY AND PARTITION --------------")
        return tiled_copy_r2s, tRS_rC, tRS_sC
    
    @cute.jit
    def make_tmem_store_and_partition_for_state(
        self,
        local_tidx,
        tmem_op_a,
        mma_tiler,
    ):
        tmem_one = tmem_op_a[((None, None), 0, 0, 0)]
        if cutlass.const_expr(mma_tiler[0] == 64):
            copy_atom_r2t = tcgen05.St16x256bOp(
                tcgen05.Repetition(1), tcgen05.Unpack.NONE,
            )
        else:
            copy_atom_r2t = tcgen05.St32x32bOp(
                tcgen05.Repetition(8), tcgen05.Unpack.NONE,
            )
        tiled_copy_r2t = tcgen05.make_tmem_copy(
            cute.make_copy_atom(copy_atom_r2t, self.io_dtype),
            tmem_one,
        )
        thr_copy_r2t = tiled_copy_r2t.get_slice(local_tidx)
        tRT_tKV16 = thr_copy_r2t.partition_D(tmem_one)
        tRT_rKV16 = cute.make_rmem_tensor(
            # cute.slice_(thr_copy_r2t.partition_S(tmem_op_a).shape, (None, None, None, None, 0)),
            # State are already picked.
            thr_copy_r2t.partition_S(tmem_one).shape,
            self.io_dtype,
        )
        print(f"tmem_op_a: {tmem_op_a}")
        print(f"tmem_one: {tmem_one}")
        print(f"tRT_tKV16: {tRT_tKV16}")
        print(f"tRT_rKV16: {tRT_rKV16}")
        return tiled_copy_r2t, tRT_tKV16, tRT_rKV16

    def tmem_load_partition_kv(self, mma_tiler, tState, local_tidx):
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            mma_tiler,
            self.o_layout,
            self.io_dtype,
            self.acc_dtype,
            mma_tiler[:2],
            use_2cta_instrs=False,
        )
        fake_sState = cute.make_tensor(
            cute.make_ptr(self.io_dtype, 0, cute.AddressSpace.smem),
            cute.dice(self.kv_mma_tiler, (1,1,None)),
        )
        return self.make_tmem_load_and_partition(
            copy_atom_t2r, tState, (None, None, 0), local_tidx, fake_sState
        )

    
    def make_tmem_load_and_partition(
        self, copy_atom_t2r, tmem_tensor, tmem_tile_coord, local_tidx, smem_tensor
    ):
        dtype = tmem_tensor.element_type
        tiled_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tmem_tensor[tmem_tile_coord])
        thr_t2r = tiled_t2r.get_slice(local_tidx)
        # Partition tmem/shared tensor for tmem load INTER1_ACC
        # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N)
        tTR_t = thr_t2r.partition_S(tmem_tensor)
        tTR_s = thr_t2r.partition_D(smem_tensor)
        # Make register fragments for tmem load INTER1_ACC
        # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N)
        tTR_r = cute.make_rmem_tensor(
            tTR_s.shape,
            dtype,
        )
        return tiled_t2r, tTR_t, tTR_r

    def tmem_store_and_partition_kv(self, local_tidx, tCrKV):
        dtype = tCrKV.element_type
        # Make tiledCopy for tensor memory store INTRA2_Q
        copy_atom_r2t = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition(8), tcgen05.Unpack.NONE),
            dtype,
        )

        tiled_r2t_kv = tcgen05.make_tmem_copy(copy_atom_r2t, tCrKV)
        thr_r2t_kv = tiled_r2t_kv.get_slice(local_tidx)

        # Partition tmem/register tensor for tensor memory store INTRA2_Q
        # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N, ...)
        tRT_rKV16 = cute.make_rmem_tensor(
            cute.slice_(thr_r2t_kv.partition_S(tCrKV).shape, (None, None, None, None, 0)),
            dtype,
        )
        # ((T2R_ATOM_V, T2R_REST_V), T2R_M, T2R_N, ..., INTERNAL_STAGE)
        tRT_tKV16 = thr_r2t_kv.partition_D(tCrKV)

        return tiled_r2t_kv, tRT_tKV16, tRT_rKV16

    @cute.jit
    def make_tmem_load_and_store_for_kv(
        self,
        local_tidx,
        tmem_acc,
        tmem_acc16,
        mma_tiler,
        acc_tile,
        kv_thr_mma,
        use_2cta_instrs=False, 
    ):
        tmem_acc_one = tmem_acc[((None, None), 0, 0, 0)]
        tmem_acc16_one = tmem_acc16[((None, None), 0, 0, 0)]

        print(f"tmem_acc: {tmem_acc}")
        print(f"tmem_acc16: {tmem_acc16}")
        print(f"tmem_acc_one: {tmem_acc_one}")
        print(f"tmem_acc16_one: {tmem_acc16_one}")

        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            mma_tiler,
            self.o_layout,
            self.io_dtype,
            self.acc_dtype,
            acc_tile,
            use_2cta_instrs,
        )
        if cutlass.const_expr(mma_tiler[0] == 64):
            copy_atom_r2t = tcgen05.St16x256bOp(
                tcgen05.Repetition(1), tcgen05.Unpack.NONE,
            )
            print(f"choose 16dpx256bx1")
        else:
            copy_atom_r2t = tcgen05.St32x32bOp(
                tcgen05.Repetition(8), tcgen05.Unpack.NONE,
            )
            print(f"choose 32dpx32bx8")

        tiled_r2t = tcgen05.make_tmem_copy(
            cute.make_copy_atom(copy_atom_r2t, self.io_dtype),
            tmem_acc16_one,
        )

        # ((V, R), TILES_M, TILES_N, STAGE)
        cKV = cute.make_identity_tensor((mma_tiler[0], mma_tiler[1]))
        tKVcKV = kv_thr_mma.partition_C(cKV)
        tileKV16likeFP32 = mma_tiler[1] // self.acc_dtype.width * self.io_dtype.width
        tKVcKV16_layout = cute.composition(
            tKVcKV.layout, cute.make_layout((mma_tiler[0], tileKV16likeFP32))
        )
        tKVcKV16 = cute.make_tensor(tKVcKV.iterator, tKVcKV16_layout)

        tiled_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tmem_acc_one)
        thr_t2r = tiled_t2r.get_slice(local_tidx)
        tTR_t = thr_t2r.partition_S(tmem_acc_one)
        tTR_c = thr_t2r.partition_D(tKVcKV)
        tTR_r = cute.make_rmem_tensor(tTR_c.shape, self.acc_dtype)

        thr_r2t = tiled_r2t.get_slice(local_tidx)
        tRT_c = thr_r2t.partition_D(tKVcKV)
        tRT_t = thr_r2t.partition_D(tmem_acc16_one)
        tRT_r = cute.make_rmem_tensor(tRT_c.shape, self.io_dtype)

        print(f"------------ MAKE TMEM LOAD AND PARTITION BEGIN --------------")
        print(f"tKVcKV: {tKVcKV}")
        print(f"tKVcKV16: {tKVcKV16}")
        print(f"LOAD tTR_r: {tTR_r}")
        print(f"LOAD tTR_t: {tTR_t}")
        print(f"LOAD tTR_r: {tTR_r}")
        print(f"STORE tRT_t: {tRT_t}")
        print(f"STORE tRT_r: {tRT_r}")
        print(f"------------ MAKE TMEM LOAD AND PARTITION END --------------")
        return tiled_t2r, tiled_r2t, tTR_t, tTR_r, tRT_t, tRT_r

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        mma_tiler: cute.Tile,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Partitions source and destination tensors for a tensor memory load.

        This method generates a tiled copy for loading accumulators from tensor
        memory and partitions the source (tensor memory) and destination
        (register) tensors accordingly.

        :param tidx: The thread index in epilogue warp groups.
        :param tAcc: The accumulator tensor to be copied and partitioned.
        :param gC_mnl: The global tensor C.
        :param epi_tile: The epilogue tiler.
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled.
        :return: A tuple containing the tiled copy for the load operation and
                 the partitioned source and destination tensors.
        """
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            mma_tiler,
            self.o_layout,
            self.io_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN)
        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None)], epi_tile
        )
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, loopM, loopN)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0)].shape, self.pv_acc_dtype
        )

        print(f"------------ EPILOG TMEM COPY AND PARTITION BEGIN --------------")
        print(f"tAcc: {tAcc}")
        print(f"tAcc_epi: {tAcc_epi}")
        print(f"gC_mnl: {gC_mnl}")
        print(f"gC_mnl_epi: {gC_mnl_epi}")
        print(f"copy_atom_t2r: {copy_atom_t2r}")
        print(f"tiled_copy_t2r: {tiled_copy_t2r}")
        print(f"thr_copy_t2r: {thr_copy_t2r}")
        print(f"tTR_tAcc: {tTR_tAcc}")
        print(f"tTR_gC: {tTR_gC}")
        print(f"tTR_rAcc: {tTR_rAcc}")
        print(f"------------ EPILOG TMEM COPY AND PARTITION END --------------")

        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    @cute.jit
    def apply_mask(
        self,
        acc_qk: cute.Tensor,
        index_qk: cute.Tensor,
        p: cute.Tensor,
        debug: bool = False,
        index_transform: cutlass.Constexpr = lambda index_q, index_k: (
            index_q,
            index_k,
        ),
    ):
        # Apply causal mask
        print(f"apply_mask acc_qk: {acc_qk}")
        print(f"apply_mask index_qk: {index_qk}")
        print(f"apply_mask p: {p}")
        for i in cutlass.range_constexpr(cute.size(acc_qk)):
            if debug:
                cute.printf("index_qk : {}", index_qk[i])
            index_q, index_k = index_transform(*index_qk[i])
            # Mask causal
            if index_q < index_k:
                acc_qk[i] = cutlass.Float32(0.0)
                p[i] = cutlass.BFloat16(0.0)
            else:
                p[i] = acc_qk[i].to(self.q_dtype)

    @cute.jit
    def make_tmem_store_and_partition(
        self, copy_atom_r2t, tmem_tensor, local_tidx
    ):
        tiled_r2t = tcgen05.make_tmem_copy(copy_atom_r2t, tmem_tensor)
        thr_r2t = tiled_r2t.get_slice(local_tidx)
        tRT_t = thr_r2t.partition_D(tmem_tensor)
        return tiled_r2t, tRT_t

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
    def local_tile_partition_for_mma_operand(
        self,
        tensor_x,
        tile_shape,
        tiled_mma,
        operand_mode,
        debug_name=None,
        no_cta_coord=False,
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
            tensor_x,
            cute.slice_(tile_shape, coord), # MK, (64, 128)
            (None, None, (hidx, bidx)) if not no_cta_coord else (None, None, None)
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

        print("===========================: {}", debug_name)
        print(f"gX: {gX}")
        print(f"tensor_x: {tensor_x}")
        print(f"thr_mma: {thr_mma}")
        print(f"tCgX: {tCgX}")
        return tCgX

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
        tCgX = self.local_tile_partition_for_mma_operand(
            tensor_x=tma_tensor_x,
            tile_shape=tile_shape,
            tiled_mma=tiled_mma,
            operand_mode=operand_mode,
            debug_name=debug_name,
        )
        # Partition shared tensor with regard to TMA
        # ((ATOM_V, REST_V), INPUT_STAGE)
        # ((ATOM_V, REST_V), TILES_N, TILES_K)
        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            tma_atom_x,
            0, # no multicast
            cute.make_layout(1),
            cute.group_modes(smem_x, 0, 3),
            cute.group_modes(tCgX, 0, 3),
        )
        print(f"tma_tensor_x: {tma_tensor_x}")
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
        # (Int32(B), Int32(S), Int32(H), Int32(D)),
        (B, S, H, D),
        stream,
    )
    compilation_time = time.time() - start_time
    print(f"Compilation time: {compilation_time:.4f} seconds")

    print(f"B, S, H, D: {(B, S, H, D)}")

    # Warmup
    for _ in range(args.warmup_iterations):
        compiled(
            q_cute.iterator,
            k_cute.iterator,
            v_cute.iterator,
            o_cute.iterator,
            decay_cute.iterator,
            (B, S, H, D),
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
            (B, S, H, D),
            stream,
        )
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    print(f"\nExecution time: {elapsed*1000/args.iterations:.2f} ms (average over {args.iterations} iterations)")
    print(f"Throughput: {(B*S*H*D*args.iterations) / (elapsed*1e9):.2f} GB/s")
    print("\nPASS")


if __name__ == "__main__":
    main()
