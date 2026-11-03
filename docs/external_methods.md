# 第三方方法、版本和许可审查

核查日期：2026-11-03。**未把第三方完整项目、ZIP、PDF 或权重加入仓库**。本次已固定 DGR/ACMIL 版本并建立独立患者适配模块，必要 MIT 声明在 docs/licenses；逐文件参考见 upstream_source_manifest.json。未知值明确标注，不编造 commit。

| 方法 | 论文与官方仓库 | 版本/许可核查 | 本地材料与下一步 |
|---|---|---|---|
| Gated-ABMIL | [ICML 2018 paper](https://arxiv.org/abs/1802.04712)；[AttentionDeepMIL](https://github.com/AMLab-Amsterdam/AttentionDeepMIL) | 本地 ZIP 没有 .git，commit 未知；本地 LICENSE 与[官方 LICENSE](https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/LICENSE) 为 MIT，版权声明为 2018 Maximilian Ilse and Jakub Tomczak | `<DOWNLOADS>/AttentionDeepMIL-master.zip` 存在；只读过 model.py/GatedAttention、README 和 LICENSE。保留原项目门控分支，官方 LeNet 编码器不能直接搬来使用。未来若复制必要代码须附原 MIT 声明 |
| DGR-MIL | [ECCV 2024 paper](https://arxiv.org/abs/2407.03575)；[official repository](https://github.com/ChongQingNoSubway/DGR-MIL) | commit `ac6257ffa768f154c52ac8b01e2086e5d3a714bf`；[LICENSE](https://github.com/ChongQingNoSubway/DGR-MIL/blob/ac6257ffa768f154c52ac8b01e2086e5d3a714bf/LICENSE)：MIT，2024 Wen | 已审查8个相关文件并独立适配；保留全局表征、交互、triplet/logdet、中心更新；短 bag 和 warmup 限制见 method_adaptations.md |
| ACMIL | [ECCV 2024 paper](https://arxiv.org/abs/2311.07125)；[official repository](https://github.com/dazhangyu123/ACMIL) | commit `e53d19c81f3892256d41962ffc4352cc91a4f834`；[LICENSE](https://github.com/dazhangyu123/ACMIL/blob/e53d19c81f3892256d41962ffc4352cc91a4f834/LICENSE)：MIT，2022 benbergner | 已审查4个相关文件；独立 ACMIL-GA 适配保留多分支、CE/attention diversity、小 bag 训练掩蔽 |
| AttriMIL | [paper](https://arxiv.org/abs/2404.00351)；[official repository](https://github.com/MedCAI/AttriMIL) | 未固定 commit；官方仓库及 README 标为 Apache-2.0；LICENSE 全文读取未成功，未来实际移植前仍需核查该文件和具体源码声明 | 本地未发现；未来核查模型和 constraints，不以患者图像序号替代病理空间坐标 |
| TDA-MIL | [MICCAI 2025 paper](https://papers.miccai.org/miccai-2025/0933-Paper2460.html)；[official repository](https://github.com/agentdr1/TDA_MIL) | 未固定 commit；[官方 README](https://github.com/agentdr1/TDA_MIL/blob/main/README.md) 现为 available soon；无已确认可用实现或许可 | 本地未发现代码包；暂不安排训练，不将空仓库标记为可运行模型 |

本地相对映射只保留 `<DOWNLOADS>` 占位符，私人绝对目录不入库。搜索覆盖原项目、使用者指定下载目录及此前常见 MIL 目录，未找到不等于证明其他磁盘不存在。

依赖 PyTorch/torchvision、NumPy、pandas、Pillow、scikit-learn、tqdm、OpenCV、PyYAML，不打包这些库。独立适配保留上列必要许可声明，未复制 fairseq 包装器，未执行官方训练流程。仓库按用户要求已公开；项目整体未擅自授予新的开源许可，第三方声明保留。
