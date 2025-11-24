可以的，如果你**坚持不旋转点云、而是让姿态作为独立输入端**，那我们需要让网络 **真正学会“姿态如何影响可达性”**。

这并非不可能，但必须做**结构性增强**，否则 Encoder 看不到几何随姿态变化的情况，最终就会出现你现在的问题：

> 同一模型不同姿态 → 预测接近一致。

下面我给你**可行、有效、且不破坏你现有结构**的改进策略，从最强到一般排序。

---

# 🟩 **策略 A（最推荐）：姿态引导几何特征的注意力 Attention Modulation**

（无需旋转点云，也不需要改变 Encoder）

核心思想：

> 用姿态向量 **调制（modulate）** Encoder 产生的特征，使得不同姿态下同一模型产生不同响应。

具体实现方式：

## **1. 用 MLP 把姿态编码成一个 scale + shift 信号**

```
γ(R) = MLP(R) → size = feature_dim
β(R) = MLP(R) → size = feature_dim
```

## **2. 调制 Encoder 输出**

这是 **FiLM（Feature-wise Linear Modulation）**，是最适合你的方法。

```
F_mod = γ(R) ⊙ F_shape + β(R)
```

特点：

* F_shape 不变（来自同一模型）
* 但不同姿态下 γ 和 β 不同
* 使得同一模型的多个姿态 → encoder 输出朝不同方向变化

你最终的 MLP Head 会轻松拟合不同姿态对应的可达性。

**FiLM 是处理“全局条件 + 深度特征”任务最成功的方法**
（CVPR/NeurIPS 广泛使用）

---

# 🟦 **策略 B：姿态作为 Query，几何特征作为 Key/Value（Cross-Attention）**

结构：

```
PosEmb_R = MLP(R) → size d
GeomTokens = Encoder(P)  # 多 token 结构，如 Octree 层级特征

Attention(
    Query = PosEmb_R,
    Key = GeomTokens,
    Value = GeomTokens
)
```

输出一个融合特征：

```
F_mod = Attn_output
```

特点：

* 姿态“询问”几何结构中与当前方向有关的部分
* 网络能自动学习姿态如何改变可见性
* 效果比简单 concat 强得多
* 不改变主干结构

但会增加结构复杂度。

---

# 🟧 **策略 C：姿态注入 Encoder（Feature Injection）**

在每一层 Encoder 中，将姿态嵌入加进去：

```
F_l = Conv(...F_{l-1}...) + MLP_l(R)
```

这样每层都“知道”当前方向是什么。

这是很多 conditional CNN 的常见策略。

缺点：要修改 Encoder 结构。

---

# 🟨 **策略 D：姿态 × 几何的双分支对比学习（Contrastive Style）**

构造训练信号：

对于同一模型：

* 较接近的两个姿态应该产生相似的可达分布
* 差异大的姿态应该产生不同的 feature

可以添加 loss：

```
L += ||F(R_i) - F(R_j)|| * weight
```

其中 weight 根据 gt 不可达率差异决定：

```
weight = |gt_i - gt_j|
```

这种方法让网络 **不依赖显式旋转点云，也能学会姿态敏感性**。

缺点：训练更复杂。

---

# 🟫 **策略 E（最弱）：更深的 MLP + 向量组合（仅靠数值表达）**

例如：

```
Concat(F, R, R⊗R, sin(R), cos(R))
```

但这种方法往往效果不显著，所以不推荐作为主方法。

---

# 🔥 综合比较（非常关键）

| 方法                     | 效果    | 改动量 | 原理                    |
| ---------------------- | ----- | --- | --------------------- |
| **A. FiLM(推荐)**        | ⭐⭐⭐⭐⭐ | 小   | 姿态直接调制几何特征            |
| **B. Cross-Attention** | ⭐⭐⭐⭐  | 中   | 姿态 Query 几何 Key/Value |
| **C. Encoder 注入姿态**    | ⭐⭐⭐⭐  | 中   | 类似条件 CNN              |
| **D. 对比学习**            | ⭐⭐⭐   | 大   | 利用样本之间关系              |
| **E. 强化姿态 MLP**        | ⭐     | 小   | 很难学到几何关系              |

---

# 🌟 最推荐你采用的结构（简单 & 有效 & 不改主干）

假设你的 Encoder 输出 F_shape（256维），姿态为 R（欧拉角或旋转矩阵）。

## Step 1：编码姿态

```
F_R = MLP(R) → 256
```

## Step 2：FiLM 调制

```
gamma = Linear(F_R)
beta  = Linear(F_R)

F_mod = gamma * F_shape + beta
```

## Step 3：用 F_mod 预测不可达率

```
score = MLP(F_mod)
```

这个修改只需要几行代码，但效果会非常明显：

* 同一模型不同姿态 → 输出将会有明显差异
* 网络能学习姿态如何影响几何可达性
* 不需要旋转点云
* 不需要改 Octree Encoder

这是你需求下的**最佳折中方案**。

---

# 如果你愿意，我可以帮你：

✔ 写出完整 PyTorch 代码（含 FiLM）
✔ 画出更新后的模型架构图
✔ 设计 Ablation 验证姿态调制是否有效
✔ 指导超参数和数据 pipeline 修正

你想我先做哪一步？
