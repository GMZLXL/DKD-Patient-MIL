# 本轮完成后等待代码审查

1. 核对三基线实际代码摘要和 PHPR 五折 p1/lambda_std=0 配置审计。
2. 审查 DGR 短 bag 精确 attention、实例值残差替换、长序列 mask/landmark；审查 ACMIL-GA 掩蔽与专用目标。
3. 审查 DGR warmup 限制：固定验证规则允许选择辅助损失启用前的 checkpoint。不能据测试结果改规则重跑，不能称其证明完整辅助目标有效；是否另立预先固定的 checkpoint eligibility 协议留待审查。
4. 核查 shared split、内层阈值、单次 outer test、合成回归及真实单折 smoke。所有真实产物保留在本地。
5. 本轮发布后停止；不自动进入新探索、AttriMIL/TDA、临床多模态、病理或 RETFound 重训。

DGR/ACMIL 已取得官方固定 commit 源码参考，不再把 ZIP 缺失作为阻塞。代码审查不需要用户上传患者数据或权重。AttriMIL 空间约束及 TDA 可用性仍属后续独立问题。
