# Linear Attention with Headwise Decay - Implementation Guide

## Overview

This document provides implementation guidance for the Chunkwise Linear Attention with Per-Head Decay kernel using CuTe DSL on NVIDIA Blackwell SM100 architecture.

## Architecture

The kernel uses a warp-specialized architecture with 512 threads (16 warps × 32 threads/warp):

- **Load Warp (1 warp)**: Handles memory data loading via TMA or async copy
- **Compute Warps (4 warps)**: Perform chunkwise attention computation with state accumulation
- **Correction Warps (4 warps)**: Apply numerical stability fixes
- **Empty/Sync Warp (1 warp)**: Manages synchronization barriers

Total: 16 warps × 32 threads = 512 threads

## Mathematical Foundation

### Per-Token Attention (No Decay)
For a single head $h$, linear attention output at position $i$:

$$O_h(i) = \frac{\sum_{j=0}^{i} K_j V_j^T Q_i}{\sum_{j=0}^{i} K_j Q_i}$$

Implemented as:
- Numerator (num): Accumulates $K_j V_j^T$ over sequence
- Denominator (denom): Accumulates $K_j$ over sequence
- Output: $O_i = \frac{Q_i^T \times \text{num}}{Q_i^T \times \text{denom} + \varepsilon}$

### With Headwise Decay
With per-head decay factor $\lambda_h \in [0, 1]$:

$$O_h(i) = \frac{\sum_{j=0}^{i} \lambda_h^{i-j} K_j V_j^T Q_i}{\sum_{j=0}^{i} \lambda_h^{i-j} K_j Q_i}$$

Implemented as exponential moving average:
- $\text{num}(t) = \lambda_h \times \text{num}(t-1) + K_t V_t^T$
- $\text{denom}(t) = \lambda_h \times \text{denom}(t-1) + K_t$

### Lightning Attention Decomposition (Chunkwise)

Sequence divided into chunks of size $L$. For chunk $c$, position $t$ within chunk:

**Intra-Chunk Attention** (local, within same chunk):
$$\text{block\_num}(t) = \lambda_h \times \text{block\_num}(t-1) + K_t V_t^T$$
$$\text{block\_denom}(t) = \lambda_h \times \text{block\_denom}(t-1) + K_t$$

**Inter-Chunk Attention** (global, accumulated state from previous chunks):
$$\text{global\_num}(c) = \lambda_h^L \times \text{global\_num}(c-1) + \text{block\_num}(c-1)$$
$$\text{global\_denom}(c) = \lambda_h^L \times \text{global\_denom}(c-1) + \text{block\_denom}(c-1)$$

**Final Output**:
$$O_h(i) = \frac{Q_i^T (\text{global\_num} + \text{block\_num})}{{Q_i^T (\text{global\_denom} + \text{block\_denom}) + \varepsilon}}$$

## Kernel Implementation Details

### Memory Layout

**Input format**: `[Batch, Sequence, Heads, Dim]`

**Reshaping for computation**: 
- Group sequence and heads together: `[Batch, (Sequence, Heads), Dim]`
- Enables efficient loading and computation across head dimension

### Data Flow

```
Global Memory (Q, K, V, decay)
         ↓
    [Load Warp]
         ↓
    Shared Memory (staging Q, K, V)
         ↓
  [Compute Warps] ← [Load Warp]
         ↓
  Register File (state accumulation)
    ├─ global_num: [H, D, D] float32
    ├─ global_denom: [H, D] float32
    ├─ block_num: [H, D, D] float32
    ├─ block_denom: [H, D] float32
    └─ decay: [H] float32
         ↓
  [Compute Warps] (iteration/reduction)
         ↓
    Shared Memory (output staging)
         ↓
  [Correction Warps] (numerical fixes)
         ↓
    Global Memory (Output O)
```

### Compute Warp Operations

Each compute warp processes a subset of the sequence with specialized responsibilities:

#### Phase 1: Load State
- Load global_num, global_denom from previous chunk
- Initialize block_num, block_denom to zero for current chunk

#### Phase 2: Intra-Chunk Recursion
For each position $t$ in current chunk (stride through positions):

1. **Load inputs**:
   - $K_t$, $V_t$ from shared memory
   - $Q_t$ from shared memory
   - $\lambda_h$ from decay vector

2. **Update block state** (via MMA operations):
   - $\text{block\_num} \leftarrow \lambda_h \times \text{block\_num} + K_t \otimes V_t^T$
   - $\text{block\_denom} \leftarrow \lambda_h \times \text{block\_denom} + K_t$
   
   Note: 
   - $K_t \otimes V_t^T$ is matrix outer product: shape `[D, D]`
   - $\lambda_h \times \text{block\_num}$: element-wise scaling
   - $K_t$: shape `[D]` vector for denominator

3. **Compute attention output**:
   - Compute $\text{total\_num} = \text{global\_num} + \text{block\_num}$: shape `[D, D]`
   - Compute $\text{total\_denom} = \text{global\_denom} + \text{block\_denom}$: shape `[D]`
   - $\text{numerator} = Q_t^T \times \text{total\_num}$: shape `[D]` (via MMA)
   - $\text{denominator} = Q_t^T \times \text{total\_denom}$: scalar (via dot product)
   - $O_t = \text{numerator} / (\text{denominator} + \varepsilon)$: element-wise division

4. **Store output**: Write $O_t$ to global memory or staging buffer

#### Phase 3: Inter-Chunk State Update
After processing entire chunk:

1. **Scale accumulated state**:
   - $\text{global\_num} \leftarrow \lambda_h^L \times \text{global\_num} + \text{block\_num}$
   - $\text{global\_denom} \leftarrow \lambda_h^L \times \text{global\_denom} + \text{block\_denom}$
   
   where $L$ = chunk_size

2. **Compute decay scaling**:
   - Precompute $\lambda_h^L$ for each head before chunk processing
   - Use in scalar multiplication with accumulator state

3. **Reset block state**:
   - $\text{block\_num} \leftarrow 0$
   - $\text{block\_denom} \leftarrow 0$

### MMA Operations Required

1. **Outer Product**: $K_t \otimes V_t^T$
   - Input shapes: $K_t \in \mathbb{R}^{D}$, $V_t \in \mathbb{R}^{D}$
   - Output shape: $\mathbb{R}^{D \times D}$
   - MMA tile: (128, 128, 32) for Blackwell (scaled for smaller D)

2. **Matrix-Vector Multiply**: $Q_t^T \times \text{block\_num}$
   - Input shapes: $Q_t \in \mathbb{R}^{D}$, $\text{block\_num} \in \mathbb{R}^{D \times D}$
   - Output shape: $\mathbb{R}^{D}$
   - MMA tile: (128, 32, 128) for Blackwell

3. **State Scaling**: $\lambda_h^L \times \text{global\_num}$
   - Element-wise scalar multiplication
   - No MMA needed

### Synchronization Barriers

```python
# Pipeline barrier between phases
cta_sync_barrier = pipeline.NamedBarrier(
    barrier_id=1,
    num_threads=512  # All threads in CTA
)

# Usage patterns:
# 1. After load warp loads data
cta_sync_barrier.arrive_and_wait()  # Compute/correction warps wait

# 2. After compute warps finish
cta_sync_barrier.arrive_and_wait()  # Correction warps can proceed

# 3. Before next iteration
cta_sync_barrier.arrive_and_wait()  # All warps synchronize
```

### Register Allocation

- **Load warp**: 32 registers (deallocated, minimal usage)
- **Compute warps**: 192 registers each (4 warps = 768 total)
  - Global state: ~64 regs (num: 32, denom: 32)
  - Block state: ~64 regs (num: 32, denom: 32)
  - Temporaries: ~64 regs (for MMA ops)
- **Correction warps**: 96 registers each (4 warps = 384 total)
  - For numerical processing and output storage
- **Empty warp**: 32 registers (deallocated)

Total: 768 + 384 = 1152 registers utilized
Available per SM: 65536, so ~1.75% usage - very efficient

### Shared Memory Usage

Key buffers needed:
1. Q_smem: `[chunk_size, head_dim]` = 64 × 64 × 4B = 16 KB
2. K_smem: `[chunk_size, head_dim]` = 64 × 64 × 4B = 16 KB
3. V_smem: `[chunk_size, head_dim]` = 64 × 64 × 4B = 16 KB
4. Synchronization barriers: < 1 KB

Total: ~48 KB (well within 192 KB limit per SM)

## Data Types and Conversions

- **IO dtype**: Float16 or BFloat16 (memory efficient)
- **Accumulation dtype**: Float32 (precision)
- **Decay dtype**: Float32 (stability)

Conversions:
1. Load Q, K, V as Float16 from global memory
2. Convert to Float32 in shared memory or registers
3. Perform all computations in Float32
4. Convert output back to Float16 for storage

## Implementation Checklist

- [ ] TMA descriptor setup for Q, K, V loading
- [ ] Load warp implementation with TMA async copy
- [ ] Compute warp state initialization (global_num, global_denom)
- [ ] Intra-chunk recursion loop
- [ ] MMA operations for outer product ($K \otimes V^T$)
- [ ] MMA operations for matrix-vector multiply ($Q^T \times \text{num}$)
- [ ] State scaling for inter-chunk updates
- [ ] Epsilon handling for numerical stability
- [ ] Correction warp implementation
- [ ] Output storage and conversion
- [ ] Synchronization barrier coordination
- [ ] Register allocation/deallocation
- [ ] Testing with reference CPU implementation

## Performance Considerations

1. **Memory bandwidth**: 
   - Q, K, V load: 3 × seq_len × head_dim reads
   - Output store: seq_len × head_dim writes
   - Decay load: num_heads reads
   - Total: O(seq_len × head_dim) bandwidth

2. **Compute intensity**:
   - MMA operations: O(head_dim²) per token
   - State updates: O(head_dim) per token
   - Output computation: O(head_dim) per token
   - Total FLOPs: O(seq_len × head_dim²)
   - Arithmetic intensity: ~head_dim FLOPs per byte

3. **Occupancy**:
   - 512 threads per block
   - Can run multiple CTAs per SM
   - Target: 2+ CTAs per SM for full utilization

4. **Expected throughput**:
   - Peak Blackwell bandwidth: 960 GB/s
   - Expected: 600-800 GB/s (80-85% of peak)
   - Latency: ~1-2 ms for [2, 1024, 8, 64] problem size

## References

- CUTLASS documentation: https://github.com/NVIDIA/cutlass
- CuTe DSL guide: https://github.com/NVIDIA/cutlass/wiki/Cute
- Lightning Attention paper: https://arxiv.org/abs/2401.04351
- Blackwell architecture: NVIDIA H200/H100 documentation

## Debugging Tips

1. **Compile with debug info**: `python setup.py build_ext --inplace -g`
2. **Check shared memory layout**: Print `cute.shape()` and `cute.stride()` of tensors
3. **Verify MMA shapes**: Ensure tile shapes match `(M, N, K)` requirements
4. **Barrier deadlocks**: Use `tight=False` in barriers during development
5. **Numerical issues**: 
   - Add epsilon early: `denom + eps` before division
   - Use float32 for intermediate accumulation
   - Monitor for NaN/Inf in output
6. **Profile with Nsight Compute**:
   - Check register pressure
   - Verify occupancy
   - Identify memory bottlenecks

## Next Steps

1. Complete TMA setup in load warp
2. Implement MMA operations in compute warps
3. Add numerical corrections in correction warps
4. Validate against CPU reference implementation
5. Performance profiling and optimization
6. Integration with larger model pipelines
