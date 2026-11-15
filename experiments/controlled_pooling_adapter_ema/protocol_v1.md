# 完整框架聚合替换协议 v1

2026-11-15冻结；开发基准 `59b776af00c0d026b69501f564f78c5efb001434`。本阶段仅开发、合成测试和 inner-only smoke，正式五折为零；后续必须另获用户授权。

## 研究组与配置来源

研究队列205名患者、703张已质控384×384 crop-only图像，K=1–9。C0 Mean、C1 Gated-ABMIL、C2 PHPR-p1共享同一图像Encoder、输入投影规格、标量分类器规格和EMA-PCAR目标。只有聚合结构变化。与旧冻结B0的35折实验分开报告，不能混表。

用户本轮明确指定历史主模型真实配置；本地 `args.json` SHA256为 `2322003d5dc28af527be359b970bd0012f8b45b352154b5c3b696977197d0150`，5个历史checkpoint内保存的args逐项对齐。这是用户指定的参考，未用外层指标重新判断哪个候选“最优”。历史源码和checkpoint均只读，不加载旧任务权重初始化新实验。

| 固定项 | 值 |
|---|---|
| Backbone | ImageNet EfficientNet-B0，1280维；原始骨干冻结，BN保持eval |
| 权重 | torchvision `efficientnet_b0_rwightman-7f5810bc.pth`；完整SHA256 `7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934` |
| Conv-Adapter | 原实现，层7、8；reduction=8、alpha=.1、dropout=.1 |
| 公共投影 | LayerNorm(1280) → Linear(256) → ReLU → Dropout(.1) |
| 公共分类器 | LayerNorm(256) → Dropout(.1) → Linear(1) |
| Gated-ABMIL | 原实现，attention_dim=128 |
| PHPR | 每类别1原型，总shape=[2,256]；lambda_proto=.1、lambda_std=0、temperature=.2；保留原fusion_norm、零初始化残差投影 |
| 患者batch / seed | 4 / 42，按组、按折重新初始化 |
| 优化器 | 原AdamW分组；Adapter LR=1e-5、head LR=1e-4、weight_decay=1e-4 |
| 调度器 | ReduceLROnPlateau(mode=max,factor=.5,patience=5)，只读inner val raw AUC |
| 预算 | 最多100 epoch；选模无改善50 epoch停止；梯度裁剪5 |
| Workers / AMP | 正式workers=4；配置AMP=true，CUDA时启用，CPU时关闭并明确记录；本次smoke workers=0 |
| 图像变换 | 直接复用历史Dataset：Resize(384)、训练翻转/小角度旋转/轻微ColorJitter、ImageNet归一化；val/test固定变换；不重复裁剪 |

输入是当前图像张量，经真实B0+Adapter可微路径生成特征，不能读取历史冻结特征缓存来训练Adapter。使用同一个本地ImageNet权重文件，不联网下载。每组相同seed的Adapter初始化相同；先构造公共Mean头，再将其投影与分类器的初始参数逐字节装入各组，防止变体模块消耗RNG导致公共分类器初值不同。结构各自的参数规模不同。

## 共享目标与EMA边界

直接继承历史 `WeightedBCEWithEMAPCARLoss`，不重新设计损失。每个患者原始logit为s、聚合特征为z，类EMA中心为C0、C1：

```
s_cal = s + .03 * (cos(z,C1) - cos(z,C0)) / 1.0
loss = weighted_BCE_with_logits(s,y)
     + .02 * mean_{positive i, negative j} softplus(.2 - (s_cal[i] - s_cal[j]))
```

pos_weight仅用本折inner train负/正患者数。训练患者随机shuffle，不根据outer test调整采样。单类batch的ranking项为0；每轮记录有效正负配对数量。训练batch先更新EMA，再计算ranking，与历史实现一致。首次缺失某类时用该训练batch总体均值初始化对应中心；此后只更新该batch存在的类别，momentum=.9。

EMA中心来自患者z，和PHPR内部可学习原型是两套独立状态；Mean/Gated同样可以提供z。只有phase=train且criterion处于train模式才更新；val/test保持只读，逐epoch检查buffer未变。每折fresh criterion；checkpoint保存模型、criterion中EMA中心及初始化标志、pos_weight、optimizer/scheduler/scaler、epoch和val依据。加载支持空动态buffer，fresh模型和fresh criterion必须精确重放raw与校准val概率。

lambda_proto是PHPR残差系数，并非额外原型监督loss；lambda_std=0时仍保留历史std分支及其requires_grad标志，其参数不影响输出，不能声称所有登记参数都有非零梯度。

## 嵌套划分、选模与阈值

保持原患者outer folds 0–4。每次仅对其余开发患者做分层8:2划分，seed42，复用已审查的 `inner_split_without_test_labels`。三个集合互斥。同一折所有组用完全相同患者列表；split函数不能读outer标签或特征。

每epoch只评估inner val；选模按 **raw AUC更大 → raw weighted BCE更小 → 更早epoch**。raw weighted BCE按患者求和除以人数，不用混入ranking的loss打破平局。无改善patience按此元组累计。best checkpoint恢复后，用其inner val原始sigmoid概率穷举F1阈值，规则复用 `src.metrics.threshold`：候选为全部val概率及0、.5、1；最大F1，平局先距.5最近，再取较小阈值。预测规则 `p >= threshold`。

主概率统一 `sigmoid(s)`。历史 `pcar_eval_score=calibrated` 与本协议不同：本协议依任务书预先改为raw主口径，校准概率只保存为次要敏感性列，不能用于主checkpoint、主阈值、主表。不得事后切换较好口径。当前不产生outer预测；未来获授权正式模式在阈值冻结后只运行一次outer test并保存预测，统计复核只读落盘结果。

## 评价与统计（预先声明，当前无正式结果）

未来每折报告ROC-AUC、PR-AUC（average precision）、Accuracy、Recall、Specificity、Precision、F1、MCC、Balanced Accuracy，及TP/FP/TN/FN、阈值、best epoch。每组五折指标汇总为算术均值±样本标准差(ddof=1)，与205患者pooled OOF九指标单独列出；pooled分类指标使用每位患者所属折的验证阈值，不再优化全体OOF阈值。保留全部方法，不依据测试表现筛选。任何新增CI或显著性分析须预先另行说明。

未来汇总文件应为 `controlled_pooling_final.csv`、`controlled_pooling_per_fold.csv`、`controlled_pooling_pooled_oof.csv`；逐患者 `controlled_pooling_oof_private.csv` 只留本地。第一阶段尚未运行这些正式汇总，也未产生正式性能结论。

## 门禁、输出与扩展结构

默认仅smoke，一次fold0；C0/C1/C2各2 epoch，内层train各类2名、val各类1名。缩小样本和workers仅为执行检查，不作为性能估计。真实smoke前自动运行7项合成测试。输出路径若已存在立即拒绝；不支持覆盖恢复或重用旧结果。门禁绑定新源码、原loss、split、metrics、ImageNet权重、参考args和图像索引SHA256。数据索引中的每张原图SHA256还会在启动时验证。

正式模式需显式 `--mode formal --fold all --formal-authorized --gate-root ...`，并通过对应组当前gate；flag仅供操作者在另获授权后使用，不构成用户授权。组与折顺序执行。每折保存实际环境、硬件、AMP状态、代码SHA256、起始Git SHA、seed、参数量、配置和私有split；真实数据/权重/预测/日志不放进Git工作树。

C3 `DGR-aggregation-only + shared loss`、C4 `ACMIL-GA-aggregation-only + shared loss` 当前仅静态可行性分析，未实现、未训练、未通过运行测试。它们剥离原生辅助目标后属于结构适配，不能称官方完整方法复现，详见 [扩展适配评估](../../docs/controlled_pooling_review.md)。

本机smoke仅CPU，未测CUDA显存、AMP数值稳定性或正式全量单epoch时间；更换执行环境后应重新做同一inner-only smoke并审查资源。本阶段到此停止，正式命令见 [运行说明](../../docs/controlled_pooling_commands.md)。
