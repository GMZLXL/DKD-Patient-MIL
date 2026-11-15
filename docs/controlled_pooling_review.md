# 聚合替换第一阶段审查报告

2026-11-15，开发基准 `59b776af00c0d026b69501f564f78c5efb001434`。C0/C1/C2开发与inner-only smoke完成；**新正式五折尚未启动**。

## 历史配置核查

初始盘点发现11个不同历史配置，当时因无法确认主模型参考而暂停。用户随后明确提供p1、lambda_proto=.1、lambda_std=0、EMA beta=.03/momentum=.9、ranking weight=.02/margin=.2、100epoch/patience50的真实配置；本地args及5个历史checkpoint内args均一致。没有用外层指标重新选择“最优”配置。旧BLOCKED草稿留作历史，新confirmed报告取代其当前状态。

直接复用 `scripts_qc/train_effb0_conv_adapter_phpr_ema_pcar_auc.py` 的Encoder、PatientPoolingHead、Dataset/collate、optimizer及EMA loss，源码SHA256 `84493765cb7333199a78432d02a8dbc3e7b4b330e08b69817b9654dbfaec7151` 未变。历史train_one_fold使用held-out fold逐轮选模，未调用。历史checkpoint没有保存完整criterion/EMA状态；新入口补全保存恢复，不改写旧checkpoint。

新增 `experiments/controlled_pooling_adapter_ema/`：core提供公共初始化、可微图像路径和EMA相位/动态buffer恢复；train提供nested统一入口；tests提供合成回归；[protocol_v1.md](../experiments/controlled_pooling_adapter_ema/protocol_v1.md)冻结全部参数。raw主概率和nested验证是任务书预先要求的协议变化，历史calibrated分数只作次要列。

## 数据、测试与smoke

205名患者/703张图像全部可读且384×384、逐图SHA256一致，患者K=1–9。五个split与原Mean保存的列表完全相同，每折inner train=131、inner val=33、outer test=41。未修改图像或元数据。

新增7项合成测试全部通过：

1. 公共投影/分类器初值逐字节一致、PHPR-p1及残差系数核对。
2. 三组K=1–9图像前后向、有限Adapter梯度、骨干冻结、eval顺序不变、NaN padding剔除及空bag拒绝。
3. 新wrapper与历史loss数值/梯度/EMA更新完全一致，val/test状态只读，eval下训练更新被拒绝。
4. 空/已初始化EMA恢复、单类batch ranking为零及fresh-fold状态。
5. 三组模型和criterion保存恢复后raw/校准预测精确一致。
6. 禁止读取outer标签/特征不影响五折split。
7. 实际新trainer在合成微型图像运行；翻转outer标签且禁止读取后，模型/EMA/checkpoint epoch/val阈值完全一致，outer推理0。

结构测试用32×32合成图像，真实smoke另用384×384原图和实际ImageNet权重。测试开发时一个fixture将label设为int，与原collate要求tensor不符，修正fixture后全套通过。没有改动真实数据或根据外层指标调整代码。门禁增加split/metrics依赖哈希后，在全新目录重新执行smoke，先前产物保留。

历史22项合成回归和独立核验器6项合成测试也通过，共35项。人工模型回归不等于重训医学实验。

真实smoke为fold0、每组2epoch，inner train每类2名、inner val每类1名，batch4、workers0、CPU两线程、effective AMP=false。每组模型+EMA恢复后val raw/校准概率精确一致，val EMA只读，outer test推理次数均为0。

| 组 | 总参数 | requires_grad参数 | Adapter参数 | head参数 | Adapter最大梯度L1 | 秒 |
|---|---:|---:|---:|---:|---:|---:|
| C0 Mean | 4,775,813 | 768,265 | 437,000 | 331,265 | 4.523831 | 5.895 |
| C1 Gated-ABMIL | 4,841,734 | 834,186 | 437,000 | 397,186 | 4.515524 | 5.765 |
| C2 PHPR-p1 | 4,909,445 | 901,897 | 437,000 | 464,897 | 4.523852 | 5.916 |

耗时包含模型构建、微型训练、验证、保存和重放，不是正式单折耗时。参数规模天然不同；C2保留lambda_std=0的历史std分支，requires_grad统计不代表所有参数都有非零梯度。环境Python3.10.11、torch2.7.0+cpu、torchvision0.22.0+cpu；CUDA显存、AMP及GPU吞吐尚未验证。

C0–C2具备当前CPU环境的代码/smoke条件，正式运行仍需本阶段审查和新授权。若迁移GPU，先在目标环境重做inner-only smoke及资源检查。

## C3/C4静态可行性

依据固定版本患者适配模块及已有上游审计，本轮没有实现或运行C3/C4。

| 方法 | 可保留的结构与共享接口 | 不能隐去的变化和问题 | 结论 |
|---|---|---|---|
| DGR | global codes、instance/global交互、cross attention、CLS的256维表示可接共同分类器和EMA；小bag沿用精确attention与padding隔离。 | 当前forward未公开患者z，需独立表示接口。原两层MLP投影、公共投影位置和global codes维度需先统一设计，避免双重投影。必须明确剥离原two-class CE、logdet diversity、triplet alignment和类别center更新；缺少原辅助约束可能造成global token冗余。 | 结构可行，未验证；需另行设计审查与梯度/小bag测试。 |
| ACMIL-GA | 5分支门控attention、训练期官方比例安全mask、attention均值形成bag z可接共享分类器和EMA；eval不mask。 | 原投影、branch/bag分类器不能直接照搬公共接口。去除branch CE与attention diversity后，新适配中未用branch_heads必须明确去除或冻结并准确计数；K=1分支attention必然相同，多分支失去diversity后可能退化。 | 结构可行，有分支退化风险，未验证；需独立适配审查。 |

未来只能命名 `DGR-aggregation-only + shared loss`、`ACMIL-GA-aggregation-only + shared loss`，不能称官方完整方法复现，不能与旧冻结特征完整目标结果混表。B组低于A组优先级，当前到静态报告为止。

## 安全与停止状态

452个受保护文件核对：451个逐字节未变，唯一有意修改是 `tools/check_release.py` 新源码/协议白名单；其余34个历史Python及全部已登记历史结果、缓存和主模型产物未变，原图另逐张校验。旧35折未重训、未覆盖。所有新输出拒绝覆盖已有fold。

发布仅含源码、合成测试、脱敏文档和source_manifest（39个Python及SHA256）。实际args、split、真实预测、权重、日志和机器路径均留在Git树之外。正式入口要求新授权flag和当前gate；本轮不执行。正式命令见 [controlled_pooling_commands.md](controlled_pooling_commands.md)。正式性能汇总、OOF及独立指标核验属于下一阶段，当前无新外层性能结论。
