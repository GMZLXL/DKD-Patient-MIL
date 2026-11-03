# 第三方方法、版本和许可审查

核查日期：2026-11-03。下表只记录来源；**未把任何第三方完整源码、ZIP、PDF 或权重加入本仓库**。此阶段没有实际使用或移植新的 upstream 版本，故 upstream commit 均未固定；未知值明确标记，不能编造 commit。未来引入源码时必须先固定 commit、检查对应文件版权并保留必要声明。

| 方法 | 论文与官方仓库 | 版本/许可核查 | 本地材料与下一步 |
|---|---|---|---|
| Gated-ABMIL | [ICML 2018 paper](https://arxiv.org/abs/1802.04712)；[AttentionDeepMIL](https://github.com/AMLab-Amsterdam/AttentionDeepMIL) | 本地 ZIP 没有 .git，commit 未知；本地 LICENSE 与[官方 LICENSE](https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/LICENSE) 为 MIT，版权声明为 2018 Maximilian Ilse and Jakub Tomczak | `<DOWNLOADS>/AttentionDeepMIL-master.zip` 存在；只读过 model.py/GatedAttention、README 和 LICENSE。保留原项目门控分支，官方 LeNet 编码器不能直接搬来使用。未来若复制必要代码须附原 MIT 声明 |
| DGR-MIL | [ECCV 2024 paper](https://arxiv.org/abs/2407.03575)；[official repository](https://github.com/ChongQingNoSubway/DGR-MIL) | 未固定 commit；已读[原 LICENSE](https://github.com/ChongQingNoSubway/DGR-MIL/blob/main/LICENSE)：MIT，版权行为 2024 Wen | 本地搜索未发现源码包；记录待核对 models/dgrmil.py、loss.py、infoNCE.py、train_DGRMIL.py。需保留全局表征、交互、对齐/多样性损失及训练中心更新 |
| ACMIL | [ECCV 2024 paper](https://arxiv.org/abs/2311.07125)；[official repository](https://github.com/dazhangyu123/ACMIL) | 未固定 commit；已读取[原 LICENSE](https://github.com/dazhangyu123/ACMIL/blob/main/LICENSE)：MIT，版权行为 2022 benbergner。其他文件及其 CLAM 来源仍需逐项核查 | 本地未发现源码包；后续核对 architecture/transformer.py 与训练入口，保留分支/目标和训练掩蔽；K=1 不能全掩蔽。本阶段未适配 |
| AttriMIL | [paper](https://arxiv.org/abs/2404.00351)；[official repository](https://github.com/MedCAI/AttriMIL) | 未固定 commit；官方仓库及 README 标为 Apache-2.0；LICENSE 全文读取未成功，未来实际移植前仍需核查该文件和具体源码声明 | 本地未发现；未来核查模型和 constraints，不以患者图像序号替代病理空间坐标 |
| TDA-MIL | [MICCAI 2025 paper](https://papers.miccai.org/miccai-2025/0933-Paper2460.html)；[official repository](https://github.com/agentdr1/TDA_MIL) | 未固定 commit；[官方 README](https://github.com/agentdr1/TDA_MIL/blob/main/README.md) 现为 available soon；无已确认可用实现或许可 | 本地未发现代码包；暂不安排训练，不将空仓库标记为可运行模型 |

本地相对映射只保留 `<DOWNLOADS>` 占位符，私人绝对目录不入库。搜索覆盖原项目、使用者指定下载目录及此前常见 MIL 目录，未找到不等于证明其他磁盘不存在。

现有研究脚本依赖 PyTorch/torchvision、NumPy、pandas、Pillow、scikit-learn、tqdm、OpenCV，仅作依赖引用，不打包第三方实现。项目整体暂未授予新的开源许可；私有仓库便于审查，后续公开前需另行审定项目归属与全部引用。
