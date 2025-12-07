# Headwise Decay Linear Attention

## 概述

Headwise decay linear attention 是一种改进的线性注意力机制，为每个注意力头引入独立的衰减因子 $\lambda_h$，实现对长依赖的更灵活建模。通过递推计算，可在 $O(n)$ 时间复杂度内完成计算。

## 基础数学公式

### 序列中第 $i$ 位置的输出

对于序列中的第 $i$ 个位置，headwise decay linear attention 的计算如下：

$$O_h(i) = \frac{\sum_{j=1}^{i} \lambda_h^{i-j} K_j V_j^T}{\sum_{j=1}^{i} \lambda_h^{i-j} K_j} \cdot Q_i$$

其中：
- $Q_i \in \mathbb{R}^d$、$K_i \in \mathbb{R}^d$、$V_i \in \mathbb{R}^{d_v}$ 分别表示位置 $i$ 的查询、键、值向量
- $\lambda_h \in [0, 1]$ 是第 $h$ 个头的衰减系数（可学习或固定参数）
- $i - j$ 是位置距离
- 分子分母都对所有位置 $j \leq i$ 进行加权累加

### 指标形式

更清晰的分量形式为：

$$O_h(i) = \frac{\sum_{j=1}^{i} \lambda_h^{i-j} K_j^T V_j Q_i}{\sum_{j=1}^{i} \lambda_h^{i-j} K_j^T Q_i}$$

## 递推计算（Token-by-Token）

为了高效地计算，引入累积状态向量：

### 状态定义

对于第 $h$ 个头，维护两个累积状态：

$$\text{num}_h(i) = \sum_{j=1}^{i} \lambda_h^{i-j} K_j V_j^T \in \mathbb{R}^{d \times d_v}$$

$$\text{denom}_h(i) = \sum_{j=1}^{i} \lambda_h^{i-j} K_j \in \mathbb{R}^d$$

### 递推关系

利用衰减的性质，可得递推公式：

$$\text{num}_h(i) = \lambda_h \cdot \text{num}_h(i-1) + K_i V_i^T$$

$$\text{denom}_h(i) = \lambda_h \cdot \text{denom}_h(i-1) + K_i$$

$$O_h(i) = \frac{Q_i^T \cdot \text{num}_h(i)}{Q_i^T \cdot \text{denom}_h(i) + \epsilon}$$

### 初始条件

$$\text{num}_h(0) = 0 \in \mathbb{R}^{d \times d_v}, \quad \text{denom}_h(0) = 0 \in \mathbb{R}^d$$

### 时间复杂度

- 时间：$O(n \cdot d^2)$（相比标准注意力的 $O(n^2 \cdot d)$ 显著降低）
- 空间：$O(h \cdot d^2)$（每个头维护两个状态）

## Chunkwise 计算（块级别）

为了进一步优化 GPU 利用率和内存局部性，采用块级别（chunk-wise）的计算方式。

### 块的定义

将长序列分割为 $C$ 个块，每块包含 $L$ 个 token（即 chunk size）：

$$N = C \times L$$

其中 $N$ 为总序列长度。

### 块内计算状态

对于第 $c$ 个块（包含 token $i_c, i_c+1, \ldots, i_c+L-1$），计算：

#### 块内部递推

在块内部，对每个位置 $t \in [0, L-1]$，位置索引为 $i = i_c + t$：

$$\text{num}_h^{(c)}(t) = \lambda_h \cdot \text{num}_h^{(c)}(t-1) + K_{i_c+t} V_{i_c+t}^T$$

$$\text{denom}_h^{(c)}(t) = \lambda_h \cdot \text{denom}_h^{(c)}(t-1) + K_{i_c+t}$$

$$O_h(i_c+t) = \frac{Q_{i_c+t}^T \cdot (\text{num}_h^{(c-1)} + \text{num}_h^{(c)}(t))}{Q_{i_c+t}^T \cdot (\text{denom}_h^{(c-1)} + \text{denom}_h^{(c)}(t)) + \epsilon}$$

其中 $\text{num}_h^{(c-1)}$ 和 $\text{denom}_h^{(c-1)}$ 是前一个块的最终状态。

#### 块间传递

块 $c$ 的最终状态为：

$$\text{num}_h^{(c)} = \sum_{t=0}^{L-1} \lambda_h^{L-1-t} K_{i_c+t} V_{i_c+t}^T$$

$$\text{denom}_h^{(c)} = \sum_{t=0}^{L-1} \lambda_h^{L-1-t} K_{i_c+t}$$

#### 块间状态累积

从块 $c-1$ 到块 $c$ 的状态传递：

$$\text{num}_h^{(c)} \leftarrow \lambda_h^L \cdot \text{num}_h^{(c-1)} + \text{num}_h^{(c)}$$

$$\text{denom}_h^{(c)} \leftarrow \lambda_h^L \cdot \text{denom}_h^{(c-1)} + \text{denom}_h^{(c)}$$

### 完整的 Chunkwise 算法

```
输入: Q, K, V (shape: [N, d]), λ_h, L (chunk size)
输出: O (shape: [N, d_v])

初始化:
  num_h ← 0_{d×d_v}
  denom_h ← 0_d
  
对每个块 c = 0, 1, ..., C-1:
  block_num ← 0_{d×d_v}
  block_denom ← 0_d
  
  对块内每个位置 t = 0, 1, ..., L-1:
    i ← c × L + t
    
    // 块内递推
    block_num ← λ_h × block_num + K_i V_i^T
    block_denom ← λ_h × block_denom + K_i
    
    // 全局状态
    local_num = num_h + block_num
    local_denom = denom_h + block_denom
    
    // 计算输出
    O_i ← (Q_i^T × local_num) / (Q_i^T × local_denom + ε)
  
  // 块结束，更新全局状态
  num_h ← λ_h^L × num_h + block_num
  denom_h ← λ_h^L × denom_h + block_denom
```

### Chunkwise 的矩阵形式

对整个块进行矩阵运算的方式：

#### 块内的矩阵递推

令 $\mathbf{K}_c$、$\mathbf{V}_c$ 分别为第 $c$ 块的键值矩阵（形状 $[L, d]$ 和 $[L, d_v]$）。

块内的累积矩阵为：

$$\text{NUM}_h^{(c)}(t) = \sum_{s=0}^{t} \lambda_h^{t-s} \mathbf{K}_c[s] \mathbf{V}_c[s]^T$$

$$\text{DENOM}_h^{(c)}(t) = \sum_{s=0}^{t} \lambda_h^{t-s} \mathbf{K}_c[s]$$

#### 块间的衰减因子

块 $c$ 对块 $c+1$ 的影响权重为 $\lambda_h^L$，对块 $c+k$ 的权重为 $\lambda_h^{kL}$。

### 内存布局与高效实现

#### 共享内存使用

- 块内 $\text{num}_h$ 和 $\text{denom}_h$ 存储在共享内存
- 块间的累积状态存储在全局内存
- 中间结果充分利用寄存器

#### 计算流程

1. **加载阶段**：加载一个块的 K、V 到共享内存
2. **计算阶段**：并行计算块内的递推关系
3. **更新阶段**：更新全局状态（块间状态累积）
4. **输出阶段**：写出 $O_i$ 到全局内存

## 复杂度分析

| 维度 | Token-by-Token | Chunkwise | 优势 |
|------|-------|-----------|------|
| **时间复杂度** | $O(n \cdot d^2)$ | $O(n \cdot d^2)$ | 相同，但 GPU 利用率更高 |
| **全局内存访问** | $O(n \cdot d)$ | $O(n \cdot d)$ | 相同 |
| **共享内存使用** | $O(d^2)$ | $O(L \cdot d)$ | Chunkwise 更灵活 |
| **计算密度** | 低（受访存限制） | 高（更多缓存复用） | Chunkwise 更优 |
| **并行度** | 有限（序列依赖） | 高（块级并行） | Chunkwise 支持更多并行 |

## 衰减系数的性质

衰减系数 $\lambda_h$ 需满足：

$$0 < \lambda_h < 1$$

- 当 $\lambda_h \to 1$ 时，注意力范围趋向于无限（所有历史 token 权重相近）
- 当 $\lambda_h \to 0$ 时，注意力集中于最近的 token（局部注意力）
- 可设置 $\lambda_h = e^{-\beta_h}$，其中 $\beta_h > 0$ 为学习的衰减速率

## 数值稳定性

为避免数值问题：

1. **分母稳定性**：添加小的 epsilon $\epsilon = 10^{-6}$
   $$O_h(i) = \frac{Q_i^T \cdot \text{num}_h(i)}{Q_i^T \cdot \text{denom}_h(i) + \epsilon}$$

2. **衰减因子稳定性**：使用对数空间计算 $\lambda_h^L = e^{L \log \lambda_h}$

3. **溢出防护**：在大块长度 $L$ 时，$\lambda_h^L$ 可能变得极小，考虑使用对数值或降低精度

## Lightning Attention 的 Chunkwise 计算表达

Lightning Attention 是一种高效的块级注意力计算方法，其核心思想是利用块间的衰减特性，将全局注意力分解为块内注意力和块间注意力。

### 块级注意力的分解

对于第 $i$ 位置（位于块 $c$ 内），其注意力可分解为两部分：

$$O_h(i) = O_h^{\text{intra}}(i) + O_h^{\text{inter}}(i)$$

其中：
- $O_h^{\text{intra}}(i)$ 为块内注意力（当前块内的贡献）
- $O_h^{\text{inter}}(i)$ 为块间注意力（历史块的贡献）

### 块内注意力（Intra-block Attention）

对于块 $c$ 内第 $t$ 个位置（$i = c \cdot L + t$），块内注意力为：

$$O_h^{\text{intra}}(i) = \frac{\sum_{s=0}^{t} \lambda_h^{t-s} K_{c,s} V_{c,s}^T}{\sum_{s=0}^{t} \lambda_h^{t-s} K_{c,s}} \cdot Q_i$$

使用块内累积状态表示：

$$\text{num}_h^{\text{intra}}(t) = \sum_{s=0}^{t} \lambda_h^{t-s} K_{c,s} V_{c,s}^T$$

$$\text{denom}_h^{\text{intra}}(t) = \sum_{s=0}^{t} \lambda_h^{t-s} K_{c,s}$$

### 块间注意力（Inter-block Attention）

块 $c$ 对块 $c+1$ 及之后的影响通过块级状态传播。块间注意力为：

$$O_h^{\text{inter}}(i) = \frac{\sum_{j=0}^{c-1} \lambda_h^{i - (j+1) \cdot L} \cdot \sum_{s=0}^{L-1} \lambda_h^{L-1-s} K_{j,s} V_{j,s}^T}{\sum_{j=0}^{c-1} \lambda_h^{i - (j+1) \cdot L} \cdot \sum_{s=0}^{L-1} \lambda_h^{L-1-s} K_{j,s}} \cdot Q_i$$

定义块级聚合状态：

$$\text{STATE}_h(c) = \left( \sum_{s=0}^{L-1} \lambda_h^{L-1-s} K_{c,s} V_{c,s}^T, \sum_{s=0}^{L-1} \lambda_h^{L-1-s} K_{c,s} \right)$$

则块间注意力可简化为：

$$O_h^{\text{inter}}(i) = \frac{Q_i^T \cdot \sum_{j=0}^{c-1} \lambda_h^{L(c-j)} \text{STATE}_h(j)[\text{num}]}{Q_i^T \cdot \sum_{j=0}^{c-1} \lambda_h^{L(c-j)} \text{STATE}_h(j)[\text{denom}]}$$

### 块级全局状态的递推

定义块 $c$ 的全局累积状态为所有历史块的加权组合：

$$\text{GLOBAL}_h(c) = \sum_{j=0}^{c-1} \lambda_h^{L(c-j)} \text{STATE}_h(j)$$

其递推关系为：

$$\text{GLOBAL}_h(c) = \lambda_h^L \cdot \text{GLOBAL}_h(c-1) + \text{STATE}_h(c-1)$$

初始条件：$\text{GLOBAL}_h(0) = 0$

### Lightning Attention 的完整表达

综合块内和块间的贡献，位置 $i = c \cdot L + t$ 的输出为：

$$O_h(i) = \frac{Q_i^T \cdot (\text{GLOBAL}_h(c)[\text{num}] + \text{num}_h^{\text{intra}}(t))}{Q_i^T \cdot (\text{GLOBAL}_h(c)[\text{denom}] + \text{denom}_h^{\text{intra}}(t)) + \epsilon}$$

其中：
- 分子中第一项为块间贡献，第二项为块内贡献
- 分母类似处理

### Chunkwise 块级计算流程

```
输入: Q, K, V (shape: [N, d]), λ_h, L (chunk size)
输出: O (shape: [N, d_v])

初始化:
  GLOBAL_num ← 0_{d×d_v}
  GLOBAL_denom ← 0_d
  
对每个块 c = 0, 1, ..., C-1:
  块级状态初始化:
    block_num ← 0_{d×d_v}
    block_denom ← 0_d
  
  对块内每个位置 t = 0, 1, ..., L-1:
    i ← c × L + t
    
    // 块内递推
    block_num ← λ_h × block_num + K_i V_i^T
    block_denom ← λ_h × block_denom + K_i
    
    // 块内注意力（完整表达）
    intra_num = block_num
    intra_denom = block_denom
    
    // 块间注意力（通过全局状态）
    global_num = GLOBAL_num
    global_denom = GLOBAL_denom
    
    // 合并块内外注意力
    total_num = global_num + intra_num
    total_denom = global_denom + intra_denom
    
    // 计算输出
    O_i ← (Q_i^T × total_num) / (Q_i^T × total_denom + ε)
  
  // 块结束，更新全局状态（用于下一个块）
  STATE_num ← block_num
  STATE_denom ← block_denom
  GLOBAL_num ← λ_h^L × GLOBAL_num + STATE_num
  GLOBAL_denom ← λ_h^L × GLOBAL_denom + STATE_denom
```

### 块级矩阵表示

定义块 $c$ 的输入矩阵：
- $\mathbf{Q}_c \in \mathbb{R}^{L \times d}$（查询）
- $\mathbf{K}_c \in \mathbb{R}^{L \times d}$（键）
- $\mathbf{V}_c \in \mathbb{R}^{L \times d_v}$（值）

块内状态的矩阵形式：

$$\text{NUM}_h^{\text{intra}}(c) \in \mathbb{R}^{L \times d \times d_v}$$
$$\text{DENOM}_h^{\text{intra}}(c) \in \mathbb{R}^{L \times d}$$

块级聚合状态（块的最终累积）：

$$\text{NUM}_h^{\text{state}}(c) = \sum_{t=0}^{L-1} \lambda_h^{L-1-t} \mathbf{K}_c[t]^T \mathbf{V}_c[t]$$

$$\text{DENOM}_h^{\text{state}}(c) = \sum_{t=0}^{L-1} \lambda_h^{L-1-t} \mathbf{K}_c[t]^T$$

全局状态递推矩阵表示：

$$\text{NUM}_h^{\text{global}}(c) = \lambda_h^L \text{NUM}_h^{\text{global}}(c-1) + \text{NUM}_h^{\text{state}}(c-1)$$

$$\text{DENOM}_h^{\text{global}}(c) = \lambda_h^L \text{DENOM}_h^{\text{global}}(c-1) + \text{DENOM}_h^{\text{state}}(c-1)$$

### 块级输出计算

块 $c$ 的所有位置输出可统一表示为：

$$\mathbf{O}_c = \text{normalize}(\mathbf{Q}_c \odot (\text{NUM}_h^{\text{global}}(c) + \text{NUM}_h^{\text{intra}}(c))) / (\mathbf{Q}_c \odot (\text{DENOM}_h^{\text{global}}(c) + \text{DENOM}_h^{\text{intra}}(c)))$$

其中 $\odot$ 表示逐行广播乘法。

### 块级并行性分析

- **块间依赖**：块 $c$ 依赖块 $c-1$ 的全局状态 $\text{GLOBAL}_h(c-1)$，构成严格的顺序依赖
- **块内并行**：块 $c$ 内的 $L$ 个位置可在获得 $\text{GLOBAL}_h(c)$ 后并行计算
- **头级并行**：不同的注意力头 $h$ 可完全独立并行处理
- **最大并行度**：$O(h \times L)$（头数乘以块大小）
