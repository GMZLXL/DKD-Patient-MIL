# DGR 参数与 Dropout 审计

审查基准：`aa6137351f5ace7e67b943f3de8587526eb3025f`。固定上游：`ChongQingNoSubway/DGR-MIL@ac6257ffa768f154c52ac8b01e2086e5d3a714bf`。下表区分**上游训练入口默认值**与**适配版实际代码值**；没有将它们解释为论文最优超参数。本次不修改历史模型或 Dropout。

| 参数/位置 | 上游训练入口或模型代码 | 已完成适配版实际值 | fullobj-v2实际值/变化 |
|---|---|---|---|
| `dropout_patch` | `train_DGRMIL.py` 默认0.1，传入 `optimizer_triple` | 共享两层投影无 Dropout（等效无随机丢弃） | 不变，无投影Dropout |
| 实例投影两处丢弃 | `optimizer_triple.drop1/drop2` 为 `Dropout1d(0.1)`；仅 instances 路径使用，global 路径绕过 | 无对应模块 | 不变 |
| `dropout_node` | `train_DGRMIL.py` 默认0.4，传到实例/全局 TransLayer 和 cross attention | 没有一个统一的 `dropout_node` CLI 参数；分别见下三行 | 不新增或调节该超参 |
| 实例 self-attention 输出 | Nystrom `to_out` 的 `Dropout1d(0.4)` | `instances.attention.out.1 = nn.Dropout(0.1)` | 不变；概率和掩蔽粒度均与上游有差异 |
| 全局 self-attention 输出 | Nystrom `to_out` 的 `Dropout1d(0.4)` | `globals.attention.out.1 = nn.Dropout(0.1)` | 不变 |
| Cross-attention 概率 | 上游包装器对 attention probabilities 做 dropout，使用入口的0.4 | `nn.MultiheadAttention(dropout=0.1)` | 不变；实际在 attention probabilities 处生效 |
| Self-attention 概率 | 上游 Nystrom 代码未在 softmax attention probabilities 处配置 Dropout | 当前精确/近似 self-attention 内也无概率 Dropout，只有输出层0.1 | 不变 |
| Cross FFN / classifier / LayerNorm | DGR 模型的这些位置无 Dropout | 当前这些位置也无 Dropout | 不变 |
| 未使用的 Mlp 类 | 上游定义了含 Dropout1d 的 Mlp，但 DGRMIL 实际图未实例化它 | 未移植未使用的类 | 不把未用类计入实际配置 |
| 隐藏维度 L | 入口 `--L` 默认512 | 256 | 不变 |
| 全局表示数 | 入口 `--num_les` 默认5；DGRMIL 构造器另有默认11 | 5 | 不变；不混用构造器默认与入口默认 |
| Cross-attention 实现 | 上游 fairseq 风格 MultiheadLinearAttention 包装器，实际 scaled dot-product softmax | PyTorch MultiheadAttention，8 heads | 不变 |
| K=1..9 attention | 上游固定 landmark 的 Nystrom 路径，原 TransLayer 未传 mask | 有效token<=128走精确 scaled dot-product，绝不声称运行了短bag近似 | 不变 |
| 实例/全局值残差 | 上游两侧为 depthwise Conv2d residual | 实例侧恒等值残差保证集合等变；全局固定token侧保留卷积 | 不变 |
| 中心 | 上游按训练bag类别更新正/负中心，momentum=.4 | 无梯度 buffer；仅 inner train objective 更新；eval不更新 | 不变 |
| 目标 | 实际上游 trainer 使用 bag BCE + 0.1 diversity + 0.1 cosine-triplet；epoch>=10启用辅助目标 | 两类CE + 同权重辅助目标；epoch>=10；Gram jitter=1e-6 | 目标、权重、启用轮次均不变 |
| 最佳checkpoint资格 | 此前患者适配入口允许warmup期入选 | 已完成5折均是 warmup-selected | 新协议仅epoch>=10、完成辅助目标更新后可入选；warmup不累计patience |

上游 DGRMIL 构造器中的 dropout_node/dropout_patch 默认都为0，而实际训练入口覆盖成0.4/0.1；不能用构造器默认值掩盖入口差异。上游 Dropout1d 与当前 nn.Dropout 的类型差异也不是只改数字：前者在三维输入上按通道丢弃，后者逐元素丢弃；实例张量的轴含义必须结合 `[1,K,L]` 解读。

`configs/adapted_models.yaml` 和 `configs/adapted_models_v2.yaml` 均是**文档/审计模板，CLI不读取**。实际运行以每折保存的 config_resolved.yaml 为准；v2 的 model.dropout_actual 由实际模块枚举得到，未给配置模板虚构驱动训练的作用。没有改变旧模型、重新搜索Dropout或根据已知测试AUC调参。
