# 项目地图

这是非破坏性、白名单式代码快照，不是完整原始工作盘的镜像。源代码相对位置和 SHA256 在 source_manifest.json；原始磁盘位置不写入仓库。

| 功能 | 文件（均在 scripts_qc/） | 核心符号 |
|---|---|---|
| ImageNet EfficientNet-B0 冻结特征 + Mean | train_whole_image_effb0_frozen_patient_mean_wbce_backup.py | ImageDataset、extract_image_features、build_patient_features、LinearPatientClassifier |
| B0、Conv-Adapter、PHPR 与门控注意力 | train_whole_image_effb0_conv_adapter_phpr.py | ConvAdapter、EfficientNetB0ConvAdapterEncoder、PatientPoolingHead、WholeImageEffB0ConvAdapterPatientPooling |
| EMA-PCAR、排序目标与风险校准 | train_effb0_conv_adapter_phpr_ema_pcar_auc.py | WeightedBCEWithAUCRankingLoss、WeightedBCEWithPCARLoss、WeightedBCEWithEMAPCARLoss |
| 患者 Dataset / collate / 五折组织 | 上述 PHPR 两个入口 | PatientWholeImageDataset、patient_collate_fn、build_patient_items、train_one_fold |
| 患者级指标 | 上述 PHPR 两个入口 | compute_metrics、safe_auc、safe_pr_auc |
| RETFound Encoder + Mean | train_retfound_patient_mean_wbce.py | build_retfound_encoder、extract_image_features、build_patient_features、train_one_fold |
| 合并标签与处理后图像元数据 | prepare_whole_image_metadata.py | find_col、build_basename_index、main |
| 历史图像裁剪 / QC 工具 | preprocess_whole_fundus_images.py | 历史预处理入口；本阶段不执行 |
| 历史患者表一致性检查 | check_patient_mil_data_qc.py | main；只在未来本地显式运行 |

## 需要审查的验证协议

历史 train_one_fold 把当前留出 fold 作为 val，并用其逐 epoch AUC 选 checkpoint。它不是独立外层 test + 内层 val 的嵌套验证。整理阶段保留该行为并清楚记录，避免伪称已修复或公平复现。

未来协议须从外层训练患者再分内层 train/val；只用内层验证早停、选阈值；外层 test 不参与任何调参。历史表的 val 字样不可直接解释为独立外层测试。

check_patient_mil_data_qc.py 保留历史相对路径常量，但这些路径只在 main 中读取，不影响导入。prepare_whole_image_metadata.py 历史上对重复 basename 取第一项，下一阶段应审查该风险；本阶段没有执行合并或改写元数据。

## 保留与未纳入

只选择眼底单模态核心和必要工具，不纳入多模态、临床、病理、旧备份、庞大的其他参数探索脚本和本地聚合实验产物。不移动任何原文件。没有新增聚合方法适配或训练入口。
