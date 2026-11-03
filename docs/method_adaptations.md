# 患者级适配与代码审查记录（2026-11-03）

## 已完成基线的实际代码已纳入

`experiments/aggregation_comparison/src/{aggregators,patient_bags,metrics}.py` 和 `scripts/train_cv.py` 是已完成 Mean / Gated-ABMIL / PHPR 五折实验实际用过的逐字节代码，不是另写的展示版。对应实际源摘要见 baseline_source_manifest.json。原 real-cache `tests/test_protocol.py` 也按原样保留在该实验目录；仓库顶层 tests/test_protocol.py 用人工合成队列测试相同 split/metrics/threshold 函数，公开测试不需要患者缓存。

历史 train_cv.py 为三种基线统一入口；保留其原样意味着它仍将 DGR/ACMIL 标为 blocked，也仍需要原缓存与 smoke 记录。它是可审查历史代码，不用于新适配方法，也未重跑。新 `scripts/train_adapted_cv.py` 仅接受 DGR/ACMIL，复用**同一** patient_bags.split、metrics.threshold 和指标函数，并保留各自必要目标。

## PHPR 配置核对

逐折读已保存 config_resolved.yaml，并检查 checkpoint 中原型张量形状；五折均为每类1个原型（共2个，shape=[2,256]）、lambda_proto=.1、lambda_std=0、temperature=.2。原始聚合类文件摘要、实际 aggregator 摘要均与运行记录一致。核对结果只含配置与形状，见 phpr_configuration_audit.json；没有患者 ID、预测或模型权重。

保留原类的 LayerNorm→1280到256投影→ReLU→Dropout(.1)、可学习原型、零初始化残差投影与最终 LayerNorm/分类器；std 模块还在，但其系数为零。聚合是 `LN(z_mean + .1*proto_proj(z_proto))`，不能写成 std 残差也参与了该次实验。未启用 Conv-Adapter、EMA-PCAR、独立校准或排序/原型辅助 loss；零初始化意味着初始时原型梯度可能为零，不能据此误删原型。

## 上游源码审查和许可

- DGR-MIL：`ChongQingNoSubway/DGR-MIL@ac6257ffa768f154c52ac8b01e2086e5d3a714bf`。审查 dgrmil.py、nystrom_attention.py、两个 cross attention 实现、train_DGRMIL.py、loss.py、infoNCE.py、LICENSE。
- ACMIL：`dazhangyu123/ACMIL@e53d19c81f3892256d41962ffc4352cc91a4f834`。审查 transformer.py、network.py、Step3_WSI_classification_ACMIL.py、LICENSE。选择官方 **ACMIL_GA** 变体，未声称实现 MHA 变体。

上游均为 MIT，必要声明在 docs/licenses/；这里只发布独立适配模块及声明，不发布原仓库完整源码或其数据。逐文件来源、固定 commit、文本参考摘要见 upstream_source_manifest.json。

## DGR-MIL-adapted

保留共享两层实例/全局表示投影、多样化可学习全局表征、可学习 CLS token、实例交互、全局 token 交互、cross attention、cross FFN 和两类分类器。全局表征数5，hidden=256（上游默认512），8 heads，landmarks=128；减小 hidden 是本次患者级计算配置，非原论文严格复现。用 PyTorch MultiheadAttention 替代实际执行缩放点积 softmax 的上游 fairseq 包装器，不引入该依赖或其源码。

### Nystrom 小 bag、填充和顺序

上游固定 landmarks=L/2，会对 K=1..9 补出大量 token；原 TransLayer 未传 mask，零填充可能进入 softmax、landmark 均值与卷积残差。传入 mask 也不足以处理全无效行 softmax 后成为均匀分布的问题。

本适配先按 valid_mask 压缩有效实例，拒绝空 bag。有效 token 数<=landmarks 时使用精确 scaled dot-product attention，不补假图，也不解大型奇异伪逆；当前 K=1..9 **全部进入该分支**。精确分支不是声称在小 bag 上运行了 Nyström 近似。长序列近似分支单独保留，并实现右侧填充、按有效计数的 landmark 均值、有效 key/query/landmark 三层 mask、全无效行归零 softmax、6次迭代伪逆及去填充。测试强制触发近似分支和空 landmark，检查有限数及 padding 不影响结果。

原实例序列上的 depthwise Conv2d 会把没有医学顺序的患者图像当作序列。本适配将**实例侧**卷积值残差改为逐实例恒等值残差，以保持集合等变性；可学习 global token 具有固定身份，其**全局侧**卷积残差保留。近似分支的分组仍可能依赖长序列顺序，不宣称超出本任务 K<=9 的集合不变性。

### 目标和中心

实际上游训练入口用 bag BCE、`-logdet` 多样性与 cosine triplet margin=1；loss.py 的 InfoNCE 不是该入口实际调用的目标，因此没有擅自将它加入本实验。本轮用形状匹配的2-logit CE（概率用 softmax 正类），保留 diversity/alignment 各0.1，从 epoch>=10 开启；Gram jitter 从1e-10改为1e-6，使用 slogdet 检查正定，避免浮点下不稳定。

正/负中心为无梯度 buffer，momentum=.4，仅由当前 **inner train** bag 的标签在 objective 中更新。forward、所有 train/val/test 预测均不更新中心；eval 模式禁止调用训练 objective。中心在计算 loss 前 clone，避免 batch 内后续更新破坏 backward 所需张量。每折重新初始化全部参数与中心。

## ACMIL-GA-adapted

保留无 bias 降维+ReLU、tanh×sigmoid 门控、多分支注意力（5分支）、各分支2类 head、平均 attention 的 bag head，以及 bag CE + branch CE + 两两 attention cosine 相似度正则（各系数1）。hidden=256、attention_dim=128，不使用图像顺序或病理坐标。

训练期候选数 `c=min(10,K_valid-1)`，实际掩蔽数 `floor(c*.6)`，从每个分支最高分 c 张中随机选，至少保留一张。K=1不掩蔽；默认 K=2 因 floor(1*.6)=0 也不掩蔽，明确记录而非伪称所有小 bag 都发生掩蔽。测试另用 mask_drop=1 覆盖最坏情况，所有 K=1..9 仍不全掩蔽。eval 自动关闭随机掩蔽。已遮蔽实例可能没有梯度，这是显式训练策略；未掩蔽的 eval 梯度测试要求每个有效实例均能影响预测。

## 统一验证策略与官方流程偏离

DGR 官方入口逐 epoch 调 test，并在 test 函数内部依据测试标签选择 ROC 阈值，随后用测试得分选 checkpoint；ACMIL 官方入口每轮 evaluate test。**这些 trainer、测试函数、阈值函数均未移植或执行。**

新入口与三种基线使用相同患者级 fold=0..4、seed42、外层训练部分分层8:2（131/33/41）、AdamW lr=.001、weight_decay=.0001、patient batch16、100 epoch、patience20、CPU2线程。固定单候选，未做测试后调参。早停：inner val AUC最大，tie时 CE最小。阈值：仅 inner val F1 最大，tie靠近.5再取较小值。训练 history无 test列；完成 checkpoint 和阈值选择后，outer test仅预测一次。

模型级/梯度/K=1..9/掩码/中心不变/防泄漏回归全部通过后，单折 smoke 只在fold0的 inner train抽8人、inner val抽4人，2epoch，检查 optimizer、保存/加载，不评估outer test；正式入口要求匹配代码摘要、缓存摘要、seed和方法的 smoke gate。所有输出使用全新目录，已有目录即报错。Mean/Gated/PHPR 及它们历史结果不重训、不覆盖。

模型和 smoke 检查证明实现可运行及协议边界，不等于证明论文指标复现或医学效能。代码审查后仍需判断上述适配是否适合论文主比较。

## 实际完成后的限制

两种适配版已完成五折；DGR 的5折最终均由固定内层验证规则选择了 epoch<10 的 checkpoint。因此这些选中模型虽包含全局表征与交互结构，但尚未接受 diversity/triplet 辅助目标训练；后续 epoch 确实启用了目标，不能把后续训练与选中的 checkpoint 混为一谈。其结果仅标为 warmup-selected 的 DGR-adapted，不能当作完整训练目标的严格复现，也不用于宣称该目标的效果。没有依据外层测试表现修改选择规则或重跑。是否另立预先固定的 checkpoint eligibility 协议应由代码审查决定。

## 本轮审查整改：独立v2，未运行正式五折

审查基准 aa6137351f5ace7e67b943f3de8587526eb3025f。以上均为历史记录，旧 acmil.py、dgr_mil.py、nystrom.py、train_adapted_cv.py 和已完成25折保持不变。

### ACMIL-GA-adapted-v2-official-mask

新增 acmil_official_ratio.py，继承旧版网络与 objective，仅独立重写掩蔽候选数。新候选 `min(10,K_valid)`，丢弃 `min(K_valid-1,floor(candidates*.6))`；仅train生效，eval不掩蔽。旧入口仍返回旧 ACMILPatient，不会静默指向v2。网络、初始化参数布局、5分支、三项损失权重、学习率均不变。

| K_valid | 旧版候选数 | 旧版丢弃数 | 新版候选数 | 新版丢弃数 |
|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 1 | 0 |
| 2 | 1 | 0 | 2 | 1 |
| 3 | 2 | 1 | 3 | 1 |
| 4 | 3 | 1 | 4 | 2 |
| 5 | 4 | 2 | 5 | 3 |
| 6 | 5 | 3 | 6 | 3 |
| 7 | 6 | 3 | 7 | 4 |
| 8 | 7 | 4 | 8 | 4 |
| 9 | 8 | 4 | 9 | 5 |

这是官方候选比例对齐的患者级小bag安全适配，仍有最终 K-1 安全上限，并非未经修改的官方模型。合成测试验证上述数量、固定seed复现、mask_drop=1最坏情况、padding剔除、有限梯度、eval顺序无关，以及同权重下新旧eval预测相同。

### DGR-MIL-adapted-fullobj-v2

模型直接复用未修改的 DGRMILPatient。新增 protocol_v2.CheckpointPolicy 和 train_adapted_cv_v2.py，固定辅助目标从epoch10启用，epoch>=10才有资格保存最佳checkpoint；warmup既不能入选，也不累计正式patience。首个合格epoch建立基准，此后仍按内层val AUC优先、CE次之，预算100、patience20、目标/中心/Dropout均不变。

checkpoint_selection.csv逐轮保存资格、是否入选、合格期stale计数、内层验证依据及辅助optimizer更新次数；checkpoint和报告保存 auxiliary_start_epoch、checkpoint_eligible_epoch、best_epoch、auxiliary_active_at_best_checkpoint。实际模型未报告辅助目标启用、没有完成辅助optimizer更新或选中epoch<10都会报错。验证仅依据内层集合；新split函数保持原分层规则并避免读取外层标签/特征。

这是在看过旧结果后提出的**事后探索性/敏感性协议**，即使未来指标改善，也不能替换旧DGR或声称事前无偏优于PHPR。本阶段只完成资格smoke，不开展正式五折。Dropout实际值与上游入口差异见 model_config_diff.md；本轮没有把它们调成上游默认值。

新CLI默认inner-only smoke；DGR smoke至少10轮，不能把2轮检查冒充完整资格验收。正式运行还要求当前v2 smoke gate及操作者在另行获得用户授权后显式传入 --formal-authorized；smoke通过不会自动触发正式运行。精确后续命令和前置条件见 review_commands_v2.md。
