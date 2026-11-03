# DKD-Patient-MIL

眼底图像单模态的患者级多实例学习研究代码。研究队列为 **205 名患者、703 张 QC 后眼底图像**，每位患者 1～9 张图像，共同预测患者级 DKD 进展高/低风险。本仓库只包含代码、配置占位模板、合成测试和文档；数据由使用者在本地提供。

当前交付为**聚合比较代码审查更新**：纳入三种已完成基线实际使用的代码，新增独立 DGR-MIL-adapted / ACMIL-GA-adapted 模块。模型及防泄漏测试、单折 smoke 通过后执行两种新方法五折；真实产物只保留在本地。三种基线不重训，历史结果不覆盖，完成后等待审查。

## 架构与代码入口

`384×384 RGB 图像 → EfficientNet-B0 特征图 → 可选 Conv-Adapter → GAP → 患者级 Mean / 门控注意力 / PHPR → 风险 logit`。

PHPR 包含患者均值主支路、可学习原型、异质性/原型残差投影及 LayerNorm。EMA-PCAR 在独立 loss/calibration 类中。RETFound + Mean 使用单独的冻结编码器入口。历史代码包含不同训练方案，不能将它们视为统一冻结编码器的公平聚合对比。

保留原脚本目录及文件名，避免整理时改变训练逻辑：

```text
configs/example.yaml               # 无真实路径的说明模板
scripts_qc/
  train_whole_image_effb0_frozen_patient_mean_wbce_backup.py
  train_whole_image_effb0_conv_adapter_phpr.py
  train_effb0_conv_adapter_phpr_ema_pcar_auc.py
  train_retfound_patient_mean_wbce.py
  prepare_whole_image_metadata.py
  preprocess_whole_fundus_images.py
  check_patient_mil_data_qc.py
tools/check_release.py              # 代码导入、语法和发布文件检查
tests/test_synthetic_heads.py       # 仅人工随机特征，CPU
tests/test_protocol.py              # 实际划分、阈值和防泄漏回归的合成测试
tests/test_adapted_models.py        # K=1..9、梯度、padding、中心和掩蔽
experiments/aggregation_comparison/
  src/aggregators.py                # 已完成三基线的原样模型代码
  src/patient_bags.py、metrics.py    # 实际划分、缓存和验证阈值
  src/adapted_models/               # 独立 DGR / ACMIL / mask-safe Nystrom
  scripts/train_cv.py               # 基线实际统一入口（存档，不重跑）
  scripts/train_adapted_cv.py        # 新方法的受 smoke gate 约束入口
  tests/test_protocol.py            # 原真实缓存协议测试源码（存档）
docs/                              # 项目地图、实验状态、许可和审查计划
```

各类和函数的对应关系见 [project_map.md](docs/project_map.md)。没有为目录美观抽取模型或改写历史训练流程。

## 环境与轻量检查

本机验证环境：Python 3.10.11、PyTorch 2.7.0+cpu、torchvision 0.22.0+cpu；其他直接依赖在 requirements.txt。CUDA 环境需自行选择相互匹配的官方 PyTorch 构建。RETFound 的运行依赖和权重必须另行提供，详见 [dependencies.md](docs/dependencies.md)。

```sh
python -m pip install -r requirements.txt
python tools/check_release.py
python -m unittest discover -s tests -v
```

默认测试只使用合成特征，不实例化图像编码器、不下载权重、不读患者文件。防泄漏回归在临时目录训练人工线性模型，证明改变外层测试标签不会改变 checkpoint 或阈值，不是医学实验。

## 本地数据和配置

训练所需元数据列：`patient_id,label,fold,processed_path`；完整项目元数据另含 `original_path`。同患者的标签和 fold 必须一致，fold 为已有患者级五折，processed_path 指向使用者本地已处理图像。不要再裁剪既有 384×384 crop-only 图像。

[example.yaml](configs/example.yaml) 仅说明字段；历史脚本不读取 YAML。查看原 CLI，不执行训练：

```sh
python scripts_qc/train_whole_image_effb0_conv_adapter_phpr.py --help
python scripts_qc/train_effb0_conv_adapter_phpr_ema_pcar_auc.py --help
python scripts_qc/train_retfound_patient_mean_wbce.py --help
```

未来运行时，显式传 `--metadata <LOCAL_METADATA_CSV> --output-dir <LOCAL_OUTPUT_DIRECTORY>`；RETFound 另需 `--retfound-root <LOCAL_RETFOUND_SOURCE_ROOT> --weight-path <LOCAL_RETFOUND_CHECKPOINT>`。除 RETFound 两个个人路径默认值改为必填参数，导出脚本只做换行规范化；原始文件不改。

## 实验状态和下一阶段

历史 EfficientNet-B0 + Mean、Conv-Adapter + PHPR + EMA-PCAR、RETFound Encoder + Mean 已有本地实验记录。统一冻结 B0 的 Mean/Gated-ABMIL/PHPR 已完成五折，本次逐字节纳入实际代码，并核对 PHPR 五折均为 p1/lambda_proto=.1/lambda_std=0/temperature=.2，checkpoint 原型 shape=[2,256]。不重训、不上传结果；来源和核对见 docs/baseline_source_manifest.json、docs/phpr_configuration_audit.json。

新方法使用相同 split/threshold 函数，保留必要目标，不复用官方逐轮测试集评估/阈值选择。K<=9 的 DGR 走精确 attention 分支；ACMIL 做小 bag 掩蔽。均为患者适配，不能声称原论文严格复现。DGR 最优 checkpoint 可能在辅助损失启用前，必须报告这一限制。详见 [method_adaptations.md](docs/method_adaptations.md)、[experiment_status.md](docs/experiment_status.md)、[review_commands.md](docs/review_commands.md)。

## 数据与第三方代码边界

禁止提交医学图像、患者 ID/标签/路径表、临床数据、特征、权重、真实 OOF/逐患者预测、日志、结果 CSV、密钥和个人绝对路径。本仓库不含这些文件，也不含下载的第三方 ZIP/PDF 或完整源码。汇总结果也需另行审查后才能加入。

官方方法的论文、固定版本和许可见 [external_methods.md](docs/external_methods.md)、docs/upstream_source_manifest.json；必要 MIT 声明在 docs/licenses/。未给项目整体擅自附加开源许可证，第三方各自许可仍适用。本轮完成后停下，等待代码审查。
