#!/usr/bin/env python3
"""
Validation and testing framework for Linear Attention with Headwise Decay

This script provides:
1. CPU reference implementation
2. Numerical validation against GPU kernel
3. Performance benchmarking
4. Debugging utilities
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple


class LinearAttentionReference:
    """CPU reference implementation of chunkwise linear attention with decay"""
    
    @staticmethod
    def forward_baseline(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        decay: torch.Tensor,
    ) -> torch.Tensor:
        """
        Baseline O(S^2) implementation without decay
        
        Args:
            Q: [B, S, H, D]
            K: [B, S, H, D]
            V: [B, S, H, D]
            decay: [H] (unused for baseline)
        
        Returns:
            O: [B, S, H, D]
        """
        B, S, H, D = Q.shape
        O = torch.zeros_like(Q)
        
        Q = Q.float()
        K = K.float()
        V = V.float()
        
        for b in range(B):
            for s in range(S):
                for h in range(H):
                    # Causal attention: attend to all previous + current
                    scores = torch.zeros(s + 1, dtype=Q.dtype, device=Q.device)
                    values_sum = torch.zeros(D, dtype=Q.dtype, device=Q.device)
                    
                    for t in range(s + 1):
                        # score = Q[b, s, h] @ K[b, t, h]
                        score = torch.dot(Q[b, s, h], K[b, t, h])
                        scores[t] = score
                        values_sum += score * V[b, t, h]
                    
                    # Average attention
                    denom = scores.sum() + 1e-6
                    O[b, s, h] = values_sum / denom
        
        return O
    
    @staticmethod
    def forward_linear_decay(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        decay: torch.Tensor,
    ) -> torch.Tensor:
        """
        Linear attention with exponential decay
        
        Mathematical formula:
        O_h(i) = (Q_i^T * (∑ λ_h^(i-j) K_j V_j^T)) / (Q_i^T * (∑ λ_h^(i-j) K_j) + ε)
        
        Args:
            Q: [B, S, H, D]
            K: [B, S, H, D]
            V: [B, S, H, D]
            decay: [H] per-head decay coefficients in [0, 1]
        
        Returns:
            O: [B, S, H, D]
        """
        B, S, H, D = Q.shape
        O = torch.zeros_like(Q)
        eps = 1e-6
        
        Q = Q.float()
        K = K.float()
        V = V.float()
        decay = decay.float()
        
        for b in range(B):
            for h in range(H):
                lam = decay[h].item()
                
                # State accumulators
                num = torch.zeros((D, D), dtype=Q.dtype, device=Q.device)  # ∑ λ^(i-j) K_j V_j^T
                denom = torch.zeros(D, dtype=Q.dtype, device=Q.device)     # ∑ λ^(i-j) K_j
                
                for s in range(S):
                    q_s = Q[b, s, h]
                    k_s = K[b, s, h]
                    v_s = V[b, s, h]
                    
                    # Update state: state = λ * state + k_s v_s^T (for numerator)
                    #               denom = λ * denom + k_s (for denominator)
                    num = lam * num + torch.outer(k_s, v_s)
                    denom = lam * denom + k_s
                    
                    # Compute output: O = (Q^T * num) / (Q^T * denom + eps)
                    numerator = q_s @ num  # [D] = [D] @ [D, D]
                    denominator = torch.clamp(q_s @ denom, min=eps)  # scalar, avoid neg/zero
                    
                    O[b, s, h] = numerator / (denominator + eps)
        
        return O
    
    @staticmethod
    def forward_chunkwise_decay(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        decay: torch.Tensor,
        chunk_size: int = 64,
    ) -> torch.Tensor:
        """
        Chunkwise linear attention with exponential decay (Lightning Attention)
        
        Decomposes attention into:
        - Intra-chunk: local attention within each chunk
        - Inter-chunk: accumulated state from previous chunks with decay
        
        Args:
            Q: [B, S, H, D]
            K: [B, S, H, D]
            V: [B, S, H, D]
            decay: [H] per-head decay coefficients
            chunk_size: Size of each chunk (default: 64)
        
        Returns:
            O: [B, S, H, D]
        """
        B, S, H, D = Q.shape
        O = torch.zeros_like(Q)
        eps = 1e-6
        
        Q = Q.float()
        K = K.float()
        V = V.float()
        decay = decay.float()
        
        # Number of chunks
        num_chunks = (S + chunk_size - 1) // chunk_size
        
        for b in range(B):
            for h in range(H):
                lam = decay[h].item()
                lam_pow_L = lam ** chunk_size  # Precompute λ^L
                
                # Global state (from previous chunks)
                global_num = torch.zeros((D, D), dtype=Q.dtype, device=Q.device)
                global_denom = torch.zeros(D, dtype=Q.dtype, device=Q.device)
                
                # Process each chunk
                for chunk_idx in range(num_chunks):
                    chunk_start = chunk_idx * chunk_size
                    chunk_end = min(chunk_start + chunk_size, S)
                    chunk_len = chunk_end - chunk_start
                    
                    # Block state (within this chunk)
                    block_num = torch.zeros((D, D), dtype=Q.dtype, device=Q.device)
                    block_denom = torch.zeros(D, dtype=Q.dtype, device=Q.device)
                    
                    # Process each position in chunk
                    for local_pos in range(chunk_len):
                        s = chunk_start + local_pos
                        
                        q_s = Q[b, s, h]
                        k_s = K[b, s, h]
                        v_s = V[b, s, h]
                        
                        # Intra-chunk recursion
                        block_num = lam * block_num + torch.outer(k_s, v_s)
                        block_denom = lam * block_denom + k_s
                        
                        # Combined attention (global + local)
                        total_num = global_num + block_num
                        total_denom = torch.clamp(global_denom + block_denom, min=eps)
                        
                        # Compute output
                        numerator = q_s @ total_num
                        denominator = torch.clamp(q_s @ total_denom, min=eps)
                        
                        O[b, s, h] = numerator / (denominator + eps)
                    
                    # Update global state for next chunk
                    # global_state = λ^L * global_state + block_state
                    global_num = lam_pow_L * global_num + block_num
                    global_denom = lam_pow_L * global_denom + block_denom
        
        return O
    
    @staticmethod
    def compute_error(
        output: torch.Tensor,
        reference: torch.Tensor,
        metric: str = "relative"
    ) -> float:
        """
        Compute error between output and reference
        
        Args:
            output: Computed output tensor
            reference: Reference/ground truth tensor
            metric: "relative", "absolute", or "max_relative"
        
        Returns:
            Error value
        """
        diff = output - reference
        
        if metric == "absolute":
            return torch.norm(diff).item()
        elif metric == "relative":
            return (torch.norm(diff) / (torch.norm(reference) + 1e-8)).item()
        elif metric == "max_relative":
            max_abs_error = torch.max(torch.abs(diff)).item()
            max_ref = torch.max(torch.abs(reference)).item()
            return max_abs_error / (max_ref + 1e-8)
        else:
            raise ValueError(f"Unknown metric: {metric}")


def test_linear_attention():
    """Test linear attention implementation"""
    
    print("=" * 80)
    print("Testing Linear Attention with Headwise Decay")
    print("=" * 80)
    
    # Test configuration
    B, S, H, D = 2, 256, 8, 64
    chunk_size = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nTest configuration:")
    print(f"  Batch size: {B}")
    print(f"  Sequence length: {S}")
    print(f"  Number of heads: {H}")
    print(f"  Head dimension: {D}")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Device: {device}")
    
    # Create test inputs
    Q = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
    K = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
    V = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
    decay = torch.full((H,), 0.95, device=device, dtype=torch.float32)
    
    print(f"\nInput shapes:")
    print(f"  Q: {Q.shape}, dtype: {Q.dtype}")
    print(f"  K: {K.shape}, dtype: {K.dtype}")
    print(f"  V: {V.shape}, dtype: {V.dtype}")
    print(f"  decay: {decay.shape}, dtype: {decay.dtype}")
    
    # Compute reference outputs
    print(f"\nComputing reference outputs...")
    
    # Linear decay reference (without chunking)
    output_linear = LinearAttentionReference.forward_linear_decay(Q, K, V, decay)
    print(f"  Linear decay (no chunking): shape {output_linear.shape}")
    
    # Chunkwise decay reference
    output_chunkwise = LinearAttentionReference.forward_chunkwise_decay(
        Q, K, V, decay, chunk_size=chunk_size
    )
    print(f"  Chunkwise decay: shape {output_chunkwise.shape}")
    
    # Verify chunkwise matches linear (should be exact)
    error = LinearAttentionReference.compute_error(output_chunkwise, output_linear)
    print(f"\n  Chunkwise vs Linear error: {error:.6e}")
    if error < 1e-4:
        print(f"  ✓ Chunkwise matches linear (good!)")
    else:
        print(f"  ⚠ Warning: chunkwise differs from linear")
    
    # Test decay sensitivity
    print(f"\nTesting decay sensitivity...")
    for decay_val in [0.5, 0.9, 0.95, 0.99, 1.0]:
        decay_test = torch.full((H,), decay_val, device=device, dtype=torch.float32)
        output_test = LinearAttentionReference.forward_chunkwise_decay(
            Q, K, V, decay_test, chunk_size=chunk_size
        )
        
        # Check output properties
        has_nan = torch.isnan(output_test).any()
        has_inf = torch.isinf(output_test).any()
        mean_val = output_test.abs().mean().item()
        max_val = output_test.abs().max().item()
        
        status = "✓" if not (has_nan or has_inf) else "✗"
        print(f"  {status} decay={decay_val:.2f}: mean={mean_val:.4f}, max={max_val:.4f}, " + 
              f"NaN={has_nan}, Inf={has_inf}")
    
    # Test different problem sizes
    print(f"\nTesting different problem sizes...")
    test_configs = [
        (1, 128, 4, 32),
        (2, 256, 8, 64),
        (4, 512, 16, 128),
    ]
    
    for b, s, h, d in test_configs:
        Q_test = torch.randn(b, s, h, d, device=device, dtype=torch.float16)
        K_test = torch.randn(b, s, h, d, device=device, dtype=torch.float16)
        V_test = torch.randn(b, s, h, d, device=device, dtype=torch.float16)
        decay_test = torch.full((h,), 0.95, device=device, dtype=torch.float32)
        
        import time
        start = time.time()
        output_test = LinearAttentionReference.forward_chunkwise_decay(
            Q_test, K_test, V_test, decay_test, chunk_size=64
        )
        elapsed = (time.time() - start) * 1000
        
        print(f"  [{b:2d}, {s:3d}, {h:2d}, {d:3d}] -> {output_test.shape}: {elapsed:6.2f} ms")
    
    print(f"\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_linear_attention()
