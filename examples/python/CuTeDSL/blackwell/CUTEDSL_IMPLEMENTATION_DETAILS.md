# CuTe DSL Linear Attention - Key Implementation Sections

This document outlines the critical code sections needed to complete the `linear_attn.py` implementation.

## 1. Load Warp Implementation

The load warp should handle efficient data loading using TMA (Tensor Memory Access).

### Pseudocode:

```python
if warp_idx == self.load_warp_id:
    cute.arch.warpgroup_reg_dealloc(32)
    
    # For each chunk of sequence:
    for chunk_idx in range(num_chunks):
        # Compute global chunk position
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, seq_len)
        chunk_len = chunk_end - chunk_start
        
        # TMA load Q for this chunk
        # Shape: [chunk_len, head_dim, num_heads, batch_size]
        q_slice = Q[chunk_start:chunk_end, :, :, :]
        cute.copy(tma_atom_q, q_slice, smem_q[...], barrier=load_barrier)
        
        # TMA load K for this chunk  
        k_slice = K[chunk_start:chunk_end, :, :, :]
        cute.copy(tma_atom_k, k_slice, smem_k[...], barrier=load_barrier)
        
        # TMA load V for this chunk
        v_slice = V[chunk_start:chunk_end, :, :, :]
        cute.copy(tma_atom_v, v_slice, smem_v[...], barrier=load_barrier)
        
        # Signal to compute warps
        cta_sync_barrier.arrive_and_wait()
```

### Key CuTe Functions:
- `cute.copy()`: Async copy operation with TMA
- `cute.shape()`: Get tensor shape
- `cute.stride()`: Get tensor stride
- `cute.layout()`: Define memory layout
- `tma_atom.prefetch()`: Prefetch TMA descriptor

## 2. Compute Warp Core Loop

The compute warps perform the main chunkwise linear attention computation.

### State Initialization:

```python
# Global state (initialized to zero, or loaded from previous block)
global_num = torch.zeros((num_heads, head_dim, head_dim), dtype=torch.float32)
global_denom = torch.zeros((num_heads, head_dim), dtype=torch.float32)

# Load decay factors once
decay = torch.zeros((num_heads,), dtype=torch.float32)
cute.copy(tma_atom_decay, decay_iter, decay_registers)

# Precompute lambda^L for each head
decay_pow_L = decay ** chunk_size  # Shape: [num_heads]
```

### Intra-Chunk Recursion:

```python
# Reset block state for this chunk
block_num = torch.zeros((num_heads, head_dim, head_dim), dtype=torch.float32)
block_denom = torch.zeros((num_heads, head_dim), dtype=torch.float32)

# Process each position in chunk
for pos in range(chunk_len):
    # Load current token's Q, K, V
    q_t = smem_q[pos]  # Shape: [head_dim, num_heads]
    k_t = smem_k[pos]  # Shape: [head_dim, num_heads]
    v_t = smem_v[pos]  # Shape: [head_dim, num_heads]
    
    # For each head
    for h in range(num_heads):
        # 1. Update block_num via MMA: block_num = lambda * block_num + K_t V_t^T
        # MMA operation: [D] x [D] -> [D, D]
        kv_outer = mma_outer_product(k_t[h], v_t[h])  # [D, D]
        block_num[h] = decay[h] * block_num[h] + kv_outer
        
        # 2. Update block_denom: block_denom = lambda * block_denom + K_t
        block_denom[h] = decay[h] * block_denom[h] + k_t[h]
        
        # 3. Compute output: O_t = (Q_t^T * total_num) / (Q_t^T * total_denom + eps)
        total_num = global_num[h] + block_num[h]    # [D, D]
        total_denom = global_denom[h] + block_denom[h]  # [D]
        
        # MMA: [D] x [D, D] -> [D]
        numerator = mma_matvec(q_t[h], total_num)   # [D]
        
        # Reduction: [D] . [D] -> scalar
        denominator = dot_product(q_t[h], total_denom) + eps
        
        # Element-wise division
        o_t = numerator / denominator  # [D]
        
        # Store output
        global_output[pos, h] = o_t

# After chunk: Update global state
global_num = decay_pow_L * global_num + block_num      # [H, D, D]
global_denom = decay_pow_L * global_denom + block_denom  # [H, D]
block_num = 0
block_denom = 0
```

### MMA Operations Needed:

#### 1. Outer Product (K ⊗ V^T):
```python
def mma_outer_product(k, v):
    """
    Compute outer product K * V^T
    Input:  k [D], v [D] 
    Output: [D, D]
    
    For Blackwell, tile shapes:
    - MMA tile: (128, 128, 32)
    - For smaller D, use partitioning
    """
    # Reshape k, v for MMA
    k_2d = cute.reshape(k, (head_dim, 1))  # [D, 1]
    v_2d = cute.reshape(v, (1, head_dim))  # [1, D]
    
    # Create tiled MMA
    tiled_mma_outer = self.make_tiled_mma_outer_product()
    
    # Partition k, v for MMA
    thr_mma_k = tiled_mma_outer.partition_a(k_2d)
    thr_mma_v = tiled_mma_outer.partition_b(v_2d)
    thr_mma_out = tiled_mma_outer.partition_c()
    
    # Perform MMA
    result = thr_mma_out
    cute.gemm(tiled_mma_outer, thr_mma_k, thr_mma_v, result)
    
    return result  # [D, D]
```

#### 2. Matrix-Vector Multiply (Q^T × matrix):
```python
def mma_matvec(q, matrix):
    """
    Compute matrix-vector product: Q^T * matrix
    Input:  q [D], matrix [D, D]
    Output: [D]
    
    Equivalent to: result[i] = sum_j Q[j] * matrix[j, i]
    """
    # Reshape for MMA
    q_2d = cute.reshape(q, (head_dim, 1))  # [D, 1]
    
    # Create tiled MMA
    tiled_mma_matvec = self.make_tiled_mma_matvec()
    
    # Partition
    thr_mma_q = tiled_mma_matvec.partition_a(q_2d)
    thr_mma_matrix = tiled_mma_matvec.partition_b(matrix)
    thr_mma_out = tiled_mma_matvec.partition_c()
    
    # Perform MMA
    result = thr_mma_out
    cute.gemm(tiled_mma_matvec, thr_mma_q, thr_mma_matrix, result)
    
    return result  # [D]
```

#### 3. Dot Product (Q · K):
```python
def dot_product(q, k):
    """
    Compute dot product: Q · K
    Input:  q [D], k [D]
    Output: scalar
    """
    result = 0.0
    for i in range(head_dim):
        result += q[i] * k[i]
    return result
```

## 3. Synchronization Patterns

### Barrier Usage:

```python
# Pattern 1: Load -> Compute synchronization
if warp_idx == self.load_warp_id:
    # Load data into shared memory
    cute.copy(...)
    # Signal done
    cta_sync_barrier.arrive_and_wait()

if warp_idx in self.compute_warp_ids:
    # Wait for load to complete
    cta_sync_barrier.wait()
    # Do computation
    ...

# Pattern 2: Compute -> Correction synchronization
if warp_idx in self.compute_warp_ids:
    # Computation
    ...
    # Signal done
    cta_sync_barrier.arrive_and_wait()

if warp_idx in self.correction_warp_ids:
    # Wait for compute
    cta_sync_barrier.wait()
    # Apply corrections
    ...
```

## 4. Correction Warp Implementation

Applies numerical stability fixes and epsilon handling.

### Pseudocode:

```python
if warp_idx in self.correction_warp_ids:
    cute.arch.warpgroup_reg_alloc(96)
    
    # Wait for compute warps
    cta_sync_barrier.wait()
    
    # Apply corrections to outputs
    for pos in range(seq_len):
        for h in range(num_heads):
            output = global_output[pos, h]
            
            # 1. Clipping: Avoid extreme values
            output = torch.clamp(output, min=-1e6, max=1e6)
            
            # 2. NaN/Inf check (optional, for debugging)
            if torch.isnan(output).any() or torch.isinf(output).any():
                output = torch.zeros_like(output)  # Or use fallback
            
            # 3. Write corrected output
            global_output[pos, h] = output
    
    # Signal done
    cta_sync_barrier.arrive_and_wait()
```

## 5. Data Type Conversions

### Input Loading:
```python
# Load as Float16 (memory efficient)
q_fp16 = load_from_global(Q)  # Float16
# Convert to Float32 for computation
q_fp32 = cute.cast(q_fp16, torch.float32)
```

### Output Storage:
```python
# Compute in Float32
output_fp32 = computation()  # Float32
# Convert to Float16 for storage
output_fp16 = cute.cast(output_fp32, torch.float16)
# Store to global memory
store_to_global(O, output_fp16)
```

## 6. Register Allocation Strategy

### Allocation Timing:

```python
# Load warp: Minimal registers needed
if warp_idx == self.load_warp_id:
    cute.arch.warpgroup_reg_dealloc(32)  # Release for others
    # Do TMA operations (use WarpGroup registers)
    cute.arch.warpgroup_reg_alloc(32)    # Reclaim before exit

# Compute warps: Heavy register usage
if warp_idx in self.compute_warp_ids:
    cute.arch.warpgroup_reg_alloc(192)
    # Register usage breakdown:
    # - State registers: 64 (global_num, global_denom, etc)
    # - MMA registers: 64-96 (for matrix operations)
    # - Temporaries: 32-48 (intermediate values)
    # Total: ~192 registers per warp

# Correction warps: Moderate register usage
if warp_idx in self.correction_warp_ids:
    cute.arch.warpgroup_reg_alloc(96)
    # Used for output processing and storage
```

## 7. Testing & Validation

### Reference CPU Implementation:

```python
def reference_linear_attention_cpu(Q, K, V, decay):
    """
    CPU reference for validation
    
    Args:
        Q: [B, S, H, D]
        K: [B, S, H, D]
        V: [B, S, H, D]
        decay: [H]
    
    Returns:
        O: [B, S, H, D]
    """
    B, S, H, D = Q.shape
    O = torch.zeros_like(Q)
    
    for b in range(B):
        for h in range(H):
            num = torch.zeros((D, D), dtype=torch.float32)
            denom = torch.zeros(D, dtype=torch.float32)
            lam = decay[h].item()
            
            for s in range(S):
                q_s = Q[b, s, h].float()
                k_s = K[b, s, h].float()
                v_s = V[b, s, h].float()
                
                # Update state
                num = lam * num + torch.outer(k_s, v_s)
                denom = lam * denom + k_s
                
                # Compute output
                total_num = num
                total_denom = denom + 1e-6
                
                numerator = q_s @ total_num
                denominator = q_s @ total_denom
                
                O[b, s, h] = numerator / (denominator + 1e-6)
    
    return O
```

### Validation Code:

```python
# Compute GPU output
O_gpu = kernel(Q, K, V, decay)

# Compute CPU reference
O_cpu = reference_linear_attention_cpu(Q, K, V, decay)

# Compare
error = torch.norm(O_gpu - O_cpu) / torch.norm(O_cpu)
print(f"Relative error: {error:.6e}")

if error < 1e-2:
    print("✓ Validation passed!")
else:
    print("✗ Validation failed!")
    print(f"Max absolute error: {torch.max(torch.abs(O_gpu - O_cpu)):.6e}")
```

## 8. Performance Optimization Tips

1. **Prefetch TMA descriptors early**: Before kernel launch
2. **Use warp-specialized barriers**: More efficient than CTA-wide synchronization
3. **Batch operations**: Process multiple chunks in parallel with different CTAs
4. **Pipeline stages**: Overlap load/compute/store phases
5. **Register reuse**: Minimize memory traffic by keeping state in registers
6. **Shared memory swizzling**: Use anti-aliasing to reduce bank conflicts

## 9. Debugging Checklist

- [ ] Verify TMA descriptor setup
- [ ] Check shared memory alignment
- [ ] Validate MMA tile shapes
- [ ] Test barrier synchronization
- [ ] Profile register usage with nsys/Nsight Compute
- [ ] Check for memory access patterns
- [ ] Verify output correctness with CPU reference
- [ ] Profile with different problem sizes

## References

- CuTe Tutorial: `CUTLASS/examples/cute`
- MMA operations: `CUTLASS/include/cute/atom/mma_atom.hpp`
- TMA documentation: `CUTLASS/include/cute/arch/copy_sm90_tma.hpp`
- Blackwell architecture: NVIDIA Hopper/Blackwell whitepapers
