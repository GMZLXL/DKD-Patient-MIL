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

只选择眼底单模态核心和必要工具，不纳入多模态、临床、病理、旧备份和真实实验产物。不移动原文件。上述 scripts_qc 为历史入口，历史选模问题没有被静默改写。

## 当前聚合比较代码

experiments/aggregation_comparison/src/aggregators.py 是三基线实际模型；patient_bags.py 和 metrics.py 是实际内层划分、缓存和阈值函数。scripts/train_cv.py 是实际用过的统一入口，原样保留，未重跑。来源摘要见 baseline_source_manifest.json。

src/adapted_models/{dgr_mil,acmil,nystrom}.py 为独立适配；scripts/train_adapted_cv.py 仅训练新方法，受模型测试和单折 smoke gate 约束，复用相同 split/threshold。scripts/verify_adapted_runs.py 只读已保存产物核对全部方法，不重新运行 outer test 模型。

顶层 tests/test_protocol.py 使用合成队列测试实际协议及 trainer 的防泄漏行为；原真实缓存测试逐字节存档在 experiments/aggregation_comparison/tests。上游版本、配置核对及差异见 upstream_source_manifest.json、phpr_configuration_audit.json、method_adaptations.md。
