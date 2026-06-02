# 实验图表意义说明

本文实验结果主要保存在：

- 图：`code/results/figures`
- 表：`code/results/tables`

这些图表可以分成六类：主结果对比、消融实验、隐私预算分析、超参数敏感性、可视化分析、通信效率分析。

## 1. 主结果对比

对应文件：

- `baselines_office_caltech.pdf`
- `baselines_cwru.pdf`
- `baselines_office_caltech.xlsx`
- `baselines_cwru.xlsx`

这些图和表回答的问题是：FedPOT 相比已有方法到底有没有提升。

表格列出 Accuracy、Macro-F1、Macro-AUC 三个指标。图中每个方法一组柱子，FedPOT 通常用红色高亮。

Office-Caltech 的结果最理想。FedPOT 明显高于 NoTransfer、FedAvg-FTL、DANN-FTL、SHOT-FTL 和 ProtoFTL，说明生成互补视角对图像特征迁移非常有效。

CWRU 的结果属于稳步提升。FedPOT 相比多数基线更好，但优势没有 Office-Caltech 那么夸张。论文里可以写成“consistently outperforms or remains competitive”，不要说成大幅碾压。

## 2. 消融实验

对应文件：

- `ablation_office_caltech.pdf`
- `ablation_cwru.pdf`
- `ablation_office_caltech.xlsx`
- `ablation_cwru.xlsx`

这些图和表回答的问题是：FedPOT 的每个模块是否真的有贡献。

消融变体包括：

- Full FedPOT：完整方法。
- w/o DP：去掉差分隐私噪声。
- w/o Partial OT：不使用部分传输，改成完整传输。
- w/o OT Reg：去掉生成器中的 OT 正则。
- w/o Filter：去掉不确定性过滤。
- w/o Soft Cond.：不用软条件，改成硬匹配条件。

左侧柱状图展示各变体的 Accuracy、Macro-F1、Macro-AUC。右侧图展示相对 Full FedPOT 的平均分数变化。

Office-Caltech 的消融能看到一些趋势，但变化幅度不大。主要可解释为：完整模型已经比较稳定，部分模块的贡献更多体现在 AUC 或鲁棒性上。

CWRU 当前结果不够理想，因为许多变体几乎不变。这通常说明该任务上模型对这些开关不够敏感，或者最终融合中有其他强信号盖过了生成视图。代码中已经调整 CWRU 的融合逻辑，后续重跑后应重点观察这两张消融图是否拉开差距。

## 3. 隐私预算分析

对应文件：

- `privacy_curve_office_caltech.pdf`
- `privacy_curve_cwru.pdf`
- `privacy_office_caltech.xlsx`
- `privacy_cwru.xlsx`

这些图和表回答的问题是：加入差分隐私噪声后，模型性能会不会明显下降。

x 轴是 privacy budget epsilon。epsilon 越小，隐私越强，噪声越大；epsilon 越大，隐私越弱，噪声越小。`No DP` 表示不加隐私噪声的上界。

Office-Caltech 的隐私曲线比较理想。从 epsilon=0.5 开始，Accuracy 基本恢复到 No-DP 水平，说明 FedPOT 对适度隐私噪声比较稳。

CWRU 的曲线比较平，说明隐私噪声对结果影响很小。这可以解释为：原型级传输本身比较稳，或者任务对 DP 噪声不敏感。

论文中可以强调：

> FedPOT retains most of the no-DP performance under practical privacy budgets, indicating that prototype-level transmission is robust to moderate DP perturbation.

## 4. 超参数敏感性

对应文件：

- `hyperparam_office_caltech.pdf`
- `hyperparam_cwru.pdf`
- `hyperparam_office_caltech.xlsx`
- `hyperparam_cwru.xlsx`

这些图和表回答的问题是：FedPOT 对关键超参数是否稳定。

分析的参数包括：

- beta：CVAE 的 KL loss 权重。
- lambda：OT regularization 权重。
- latent dimension：CVAE 潜变量维度。

Office-Caltech 的图比较有意义。默认设置 `beta=2.0`、`lambda=0.01`、`latent_dim=128` 表现靠前，说明参数选择合理。

CWRU 的图比较平。beta 和 lambda 改变后性能几乎不变，latent_dim=64 略好。这张图更适合解释为“模型对超参数比较鲁棒”，而不是“强敏感性分析”。

## 5. OT 对齐热力图

对应文件：

- `ot_heatmap_office_caltech.pdf`
- `ot_heatmap_cwru.pdf`

这些图回答的问题是：Partial OT 是否学到了合理的跨侧语义对齐。

图中：

- 行表示 t-side clusters。
- 列表示 d-side classes。
- 颜色越深表示 transport mass 或 alignment probability 越高。
- 红框标出每一行最强的匹配。

Office-Caltech 的热力图最清楚，基本呈现近似一对一匹配，说明 OT 能把目标侧聚类和源侧类别对应起来。

CWRU 的热力图也可用，类别更少，结构更简单。适合说明模型能找到主要故障类别的对应关系。

## 6. t-SNE 可视化

对应文件：

- `tsne_office_caltech.pdf`
- `tsne_cwru.pdf`

这些图回答的问题是：生成特征是否在低维空间中形成有意义的类别结构。

每张图通常包含：

- t-side original features
- d-side original features
- FedPOT generated features

Office-Caltech 的 t-SNE 可用，生成特征中多个类别有明显聚集，说明生成视图带有类别语义。

CWRU 的 t-SNE 一般，生成特征有结构，但类间混杂仍存在。论文中可以谨慎描述为“improves semantic grouping”或“forms more structured clusters”，不要说完全分离。

## 7. 融合特征可视化

对应文件：

- `fusion_embedding_office_caltech.pdf`
- `fusion_embedding_cwru.pdf`

这些图回答的问题是：t-side 原始特征、生成 d-view 特征、融合特征之间有什么变化。

通常三列分别是：

- t-side view
- Generated d-view
- FedPOT fused view

理想情况下，融合视图应该比单独 t-side 或 generated d-view 更容易区分类别。

Office-Caltech 的融合图能看到类别分离增强，但 generated d-view 有些点簇呈条带状，视觉上略显人工。建议正文可以放 t-SNE 和 OT heatmap，fusion embedding 可放附录。

CWRU 的融合图可用，但类别仍有混合。适合辅助说明，而不是作为最强证据。

## 8. 校准曲线

对应文件：

- `calibration_office_caltech.pdf`
- `calibration_cwru.pdf`

这些图回答的问题是：模型预测置信度是否可靠。

x 轴是 confidence，y 轴是 empirical accuracy。曲线越接近对角线，说明模型越校准。

当前 Office-Caltech 的校准曲线波动较大，主要原因是测试样本较少，每个 confidence bin 内样本数量有限。它可以作为辅助分析，但不建议作为强结论图。

CWRU 的校准图稍稳定，但仍有跳动。代码中已经把 calibration 计算改成和正式评估一致的 fused logits，并减少 bin 数，重跑后会更稳一些。

## 9. 语义检索可视化

对应文件：

- `semantic_retrieval_office_caltech.pdf`

这张图回答的问题是：生成的互补特征是否能改善近邻检索的语义一致性。

每一行通常包含：

- Query：查询图像。
- t-only NN：只用 t-side 特征找到的最近邻。
- FedPOT NN：使用 t-side + generated d-view 后找到的最近邻。

如果 FedPOT NN 的类别更接近 Query，就说明生成特征补充了有用语义。

这张图直观性强，适合放在可视化实验中。不过文件较大、排版容易拥挤，代码中已经调整了行距和图片尺寸。

## 10. CWRU 生成特征曲线

对应文件：

- `cwru_generated_feature_profiles.pdf`

这张图回答的问题是：FedPOT 生成的 d-view 特征是否接近真实 source d-view 的统计形态。

每个子图对应一个故障类别。红色通常表示 generated d-view，蓝色虚线表示 source d-view。

如果两条曲线的走势相近，可以说明生成器不是随机造特征，而是在学习 d-side 的类别相关模式。

这张图适合 CWRU，因为振动信号特征不像图像那样能直接展示原图，用特征曲线更自然。

## 11. 通信效率分析

对应文件：

- `comm_analysis.pdf`
- `comm_analysis.xlsx`
- `comm_analysis.json`

这些图和表回答的问题是：FedPOT 的通信成本是否更低。

FedPOT 只传输 t-side clusters 和 d-side class prototypes 相关的原型向量，通信量近似为：

```text
O((K + C) * d)
```

其中 K 是类别数，C 是聚类数，d 是原型维度。

相比之下，FedAvg-FTL、DANN-FTL、SHOT-FTL 通常需要传输模型参数，通信量与参数规模 P 相关，即：

```text
O(P)
```

因此通信图中 FedPOT 明显低于模型传输类方法。ProtoFTL 通信量也低，但它缺少 OT 对齐和条件生成能力。

这张图比较理想，适合放正文，用来支撑“communication-efficient”的主张。

## 12. 表格在论文中的用途

`baselines_*.xlsx`：主结果表，适合整理成论文 Table 1。

`ablation_*.xlsx`：消融表，适合整理成论文 Table 2 或附录表。

`privacy_*.xlsx`：隐私预算分析，适合配合隐私曲线使用。

`hyperparam_*.xlsx`：超参数分析，适合附录或实验设置说明。

`comm_analysis.xlsx`：通信成本表，适合和通信效率图一起放正文。

## 13. 当前最推荐正文展示的图

优先推荐放正文：

- `baselines_office_caltech.pdf`
- `baselines_cwru.pdf`
- `ot_heatmap_office_caltech.pdf`
- `privacy_curve_office_caltech.pdf`
- `hyperparam_office_caltech.pdf`
- `comm_analysis.pdf`
- `semantic_retrieval_office_caltech.pdf`

谨慎放正文或放附录：

- `ablation_cwru.pdf`
- `calibration_office_caltech.pdf`
- `fusion_embedding_office_caltech.pdf`
- `fusion_embedding_cwru.pdf`

原因是这些图目前要么变化不明显，要么视觉上不够稳，适合作为补充材料，而不是主证据。

## 14. 写论文时的总体解释

可以把所有实验串成下面这条逻辑：

1. 主结果证明 FedPOT 比现有基线更准。
2. OT heatmap 和 t-SNE 证明 FedPOT 学到了跨侧语义对齐。
3. 语义检索和生成特征曲线证明生成的互补视图有实际语义。
4. 隐私曲线证明加入 DP 后性能仍然稳定。
5. 通信分析证明 FedPOT 比传模型参数的方法更省通信。
6. 超参数实验说明方法对关键参数不脆弱。

白话来说就是：

> FedPOT 不只是分数高，它还能解释为什么高：原型对齐是清楚的，生成特征是有语义的，隐私噪声下性能是稳的，通信量也是省的。

