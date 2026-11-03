# 当前提交验证记录

- 三基线实际用过的9个源码文件逐字节纳入，原真实缓存测试也存档；顶层公开测试用合成数据复核相同协议函数。
- PHPR 五折配置与 checkpoint 原型形状一致：每类1原型、lambda_proto=.1、lambda_std=0、temperature=.2；原类与 aggregator 摘要匹配运行记录。
- 11项公开测试通过，覆盖 K=1..9、前后向有限数、参数更新、所有有效实例推理梯度、无效 padding 零梯度/不影响输出、ACMIL 最坏掩蔽、DGR 中心更新边界、Nystrom 近似路径空 landmark、实际划分和阈值、防外层标签影响选模。
- DGR 和 ACMIL 分别完成 fold0 内层 train 8人 / val 4人、2epoch 的真实缓存 smoke，训练和保存/加载通过；外层 test 评估次数0。smoke 使用独立目录和代码/缓存/seed摘要 gate。
- gate 通过后，两种适配版各完成五折；只读检查25折（含原三基线）和1025条本地OOF，确认患者划分相同、每方法每患者一次、内层 checkpoint/阈值选择一致，真实预测只保留本地。每个正式折 outer test 评估1次，核验阶段新增模型test评估0次。
- 141个历史基线产物的摘要保持不变；没有重训 Mean/Gated-ABMIL/PHPR，没有覆盖历史结果。
- DGR 的5折选中 checkpoint 均早于辅助目标启用。这是已记录的科学限制，未在查看测试表现后更改协议；详见 method_adaptations.md，不可称为完整DGR目标的验证结果。
- 仓库安全检查覆盖实际暂存内容；仅允许源代码、配置模板、文档和经过限定的非敏感源码/配置摘要JSON。数据、缓存、权重、真实预测、CSV、日志和原始官方项目不入库。

阶段0原记录保留在 release_validation.md，描述当时的历史检查范围，不代表本次新增适配的状态。当前状态以本文件和 experiment_status.md 为准。本轮完成后等待代码审查。
