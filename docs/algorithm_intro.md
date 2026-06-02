# FedPOT 算法介绍

## 1. 问题背景

FedPOT 面向的是一种异构联邦迁移学习场景：两端客户端拥有同一任务相关的数据，但它们的特征视角不同、特征维度不同，并且不能直接共享原始数据。

在本文实验中，可以把两端理解成：

- t-side：目标侧，拥有目标域样本的不完整特征，用于最终分类。
- d-side：源侧或补充侧，拥有另一组互补特征，可提供额外语义信息。
- 目标：在不传输原始样本的前提下，让 t-side 借助 d-side 的语义结构，提升目标侧下游分类性能。

传统做法往往直接传模型参数、传特征映射，或者只用简单原型匹配。这些方法要么通信量大，要么难以处理 t-side 和 d-side 特征空间不一致的问题。FedPOT 的核心思路是：只交换经过隐私保护的原型，通过部分最优传输建立跨侧语义对齐，再用条件生成模型为 t-side 补出 d-side 风格的互补特征。

## 2. 整体流程

FedPOT 可以拆成五个阶段：

1. 原型提取与隐私保护
2. 部分最优传输语义对齐
3. 传输条件驱动的 CVAE 特征生成
4. 不确定性过滤
5. 下游分类与融合评估

整体逻辑是：

```text
t-side features        d-side features
      |                      |
  t-side prototypes      d-side class prototypes
      |                      |
      +---- DP noise / prototype transmission ----+
                                                  |
                         Partial OT alignment
                                  |
                       transport-conditioned CVAE
                                  |
                generated complementary d-view features
                                  |
                    t-view + generated d-view fusion
                                  |
                         downstream classifier
```

## 3. 模块一：原型提取与差分隐私

FedPOT 不直接传输样本级特征，而是传输原型级统计量。

在 d-side，算法根据类别标签计算每个类别的特征均值，得到类别原型。  
在 t-side，算法对目标侧特征做聚类，得到若干目标侧聚类原型。

为了降低原型泄露原始数据分布的风险，原型在传输前会加入差分隐私噪声。这样传输内容从“样本级数据”降到了“带噪声的语义中心”，通信量和隐私风险都更低。

这一模块的作用是：

- 用少量原型概括两侧语义结构。
- 避免传输原始样本或完整模型。
- 为后续 OT 对齐提供输入。

## 4. 模块二：部分最优传输语义对齐

t-side 的聚类原型和 d-side 的类别原型处在不同视角下，不能简单一一对应。FedPOT 使用 Partial Optimal Transport 计算二者之间的软对齐关系。

这里的 OT 矩阵可以理解为：

- 每一行对应一个 t-side cluster。
- 每一列对应一个 d-side class prototype。
- 矩阵值表示这个 t-side cluster 应该从哪些 d-side 类别原型中吸收语义信息。

使用“部分”最优传输的原因是：不是所有目标侧聚类都一定能被源侧类别完美解释。Partial OT 允许只传输一部分质量，从而降低错误对齐对后续生成的影响。

这一模块的意义是：

- 建立 t-side cluster 和 d-side class 之间的语义桥梁。
- 用软对齐保留不确定性，而不是强行硬匹配。
- 为生成器提供跨侧条件信息。

## 5. 模块三：条件 CVAE 生成互补特征

得到 OT 对齐后，FedPOT 会为每个 t-side 样本构造一个 transport condition。这个条件通常是 d-side 原型的加权组合，代表“这个样本在 d-side 视角下可能对应的语义补充”。

然后算法训练一个 Conditional VAE：

- 输入：t-side 原始特征。
- 条件：由 OT 矩阵构造的 d-side 语义条件。
- 输出：生成的 d-side 风格互补特征。

生成结果不是图像或原始数据，而是特征层面的补全。最终每个样本会拥有：

```text
fused feature = [t-side original feature ; generated d-side complementary feature]
```

这一模块的核心贡献是：把跨侧原型对齐转化为样本级特征增强，使目标侧分类器能用到更完整的语义信息。

## 6. 模块四：不确定性过滤

生成特征可能存在噪声。FedPOT 引入不确定性过滤，主要考虑两类信号：

- reconstruction uncertainty：生成器重构是否稳定。
- semantic uncertainty：生成特征和条件原型是否语义一致。

过滤器不会简单删除所有低质量样本，而是通过 sample weight 降低不可靠生成样本在训练中的影响。

这一模块的作用是：

- 防止错误生成特征干扰分类器。
- 提高增强特征的可信度。
- 保留一定样本覆盖率，避免过滤过猛。

## 7. 模块五：下游分类与融合

最后，FedPOT 训练两个分类视角：

- baseline classifier：只使用 t-side 原始特征。
- FedPOT classifier：使用 t-side 原始特征和生成的 d-side 互补特征拼接后的增强特征。

评估时，算法会融合 baseline logits 和 FedPOT logits。这样做的直观原因是：原始 t-side 特征通常比较稳定，而生成特征提供额外信息但也可能有噪声，融合能在稳定性和增强效果之间折中。

对于 CWRU 数据集，代码中还加入了 prototype-alignment view 作为辅助信号，但它只应作为辅助，而不能盖过 FedPOT 生成视图，否则消融实验会看不出各模块的真实影响。

## 8. 算法优势

FedPOT 的优势可以总结成四点：

- 隐私友好：只传带噪原型，不传原始样本。
- 通信高效：通信量与类别数、聚类数和特征维度相关，而不是与模型参数量或样本数强绑定。
- 适合异构特征：通过 OT 在语义层面对齐两侧视角，不要求两侧特征空间完全一致。
- 可解释性较强：OT heatmap、t-SNE、生成特征曲线都能展示跨侧对齐和特征补全过程。

## 9. 和基线方法的区别

NoTransfer 只使用目标侧特征，不借助 d-side 信息。

FedAvg-FTL、DANN-FTL、SHOT-FTL 更偏向模型或特征迁移，通信量通常更大，并且对异构特征适配不够自然。

ProtoFTL 也使用原型，但主要依赖最近邻或简单原型匹配，没有 Partial OT 的软语义对齐，也没有 CVAE 生成互补视图。

FedPOT 的关键区别是把“原型传输、OT 语义对齐、条件生成、过滤增强”串成完整闭环。

## 10. 论文中可以强调的主线

可以把 FedPOT 的论文叙事写成：

> FedPOT first compresses heterogeneous private data into differentially private prototypes, then aligns target clusters and source-side semantic prototypes with partial optimal transport. The learned transport plan is further used as a soft condition to generate complementary target-side features, enabling privacy-preserving and communication-efficient federated transfer across heterogeneous feature spaces.

中文白话版就是：

> FedPOT 不直接搬数据，也不粗暴搬模型，而是先把两边数据压缩成带隐私保护的语义原型，再用部分最优传输找到两边语义上的对应关系，最后根据这种对应关系给目标侧样本生成缺失视角的互补特征，从而提升分类效果。

