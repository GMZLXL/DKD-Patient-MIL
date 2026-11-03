# 实验状态（本轮更新后，等待代码审查）

| 工作 | 状态 | 与下一阶段的关系 |
|---|---|---|
| EfficientNet-B0 + Mean 历史实验 | 已有本地实验记录 | 历史验证方案不能替代新嵌套比较 |
| Conv-Adapter + PHPR + EMA-PCAR | 已有本地实验记录 | 联合适配和校准不等于冻结编码器 PHPR |
| RETFound Encoder + Mean | 已完成历史编码器对比 | 不在本阶段重训；不是统一 B0 聚合头比较 |
| 冻结 B0 的 Mean / Gated-ABMIL / PHPR | 已完成五折，本轮未重训 | 9个实际源码文件逐字节纳入；141个历史产物摘要未改变；不发布真实产物 |
| Gated-ABMIL | 实际模型及统一训练入口已纳入 | 门控公式、原内层划分、验证阈值和实际测试源码可供审查 |
| DGR-MIL-adapted | 完成模型/梯度/K=1..9/单折 smoke 后，完成五折 | 使用独立新目录。**5/5折验证最优 checkpoint 均早于辅助损失启用，不可视为完整辅助目标已验证或严格原方法复现** |
| ACMIL-GA-adapted | 完成模型/梯度/K=1..9/单折 smoke 后，完成五折 | 保留分支/attention diversity/训练掩蔽；每折测试只评估一次；不是 MHA 变体 |
| PHPR 配置核对 | 五折配置与 checkpoint 形状一致 | p1、lambda_proto=.1、lambda_std=0、temperature=.2；没有 Adapter/EMA-PCAR |
| ACMIL-GA-adapted-v2-official-mask | 22项合成测试中的对应测试及真实inner-only 2轮smoke通过；**正式五折0折** | 独立版本，候选数按K_valid；旧ACMIL实现及五折不变 |
| DGR-MIL-adapted-fullobj-v2 | 合格checkpoint回归及真实inner-only 10轮smoke通过；**正式五折0折** | epoch>=10才选模/累计patience；真实smoke best_epoch=10且有辅助更新；不替换旧DGR |
| AttriMIL | 扩展性评估待开展 | 无天然病理空间坐标，不能用图像序号替代 |
| TDA-MIL | 暂无可用实现 | 官方当前 README 标注 available soon，不安排训练 |

本地只读核验25折、每方法205条OOF（合计1025）；各方法患者划分完全一致，按已保存内层验证记录核对最佳 checkpoint 和阈值。验证不重复运行外层测试模型。新结果只写到全新本地目录，原历史结果不变。

公开仓库仅发布源代码、非敏感配置核查和运行完成状态；没有真实患者数据/预测、缓存、权重、结果CSV或训练日志。DGR 的 warmup 限制、不同 head 参数量、单种子/单候选预算都需要审查，不能宣称严格论文复现或显著优于。完成此提交后停止，等待审查。

本轮审查整改以aa613735为基准，正式实验总数仍是25折；234个历史产物/缓存文件、27个历史Python源码的前后SHA256一致。两个v2是事后敏感性分析，仅完成模型/资格/隔离测试及单折smoke，不产生新的outer test结果。本阶段明确禁止正式五折，后续必须再次授权。
