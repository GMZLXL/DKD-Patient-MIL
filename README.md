# DKD-Patient-MIL

眼底图像单模态的患者级多实例学习研究代码。研究队列为 **205 名患者、703 张 QC 后眼底图像**，每位患者 1～9 张图像，共同预测患者级 DKD 进展高/低风险。本仓库只包含代码、配置占位模板、合成测试和文档；数据由使用者在本地提供。

当前工作为**CPU执行优化验证**：用户授权的原正式任务在C0 fold0完成37轮后，按用户选择停止并完整保留，尚无完成的外层折。新增独立CPU后端，复用全部模型/损失/epoch/选模代码，仅调整worker生命周期与验证加载；结果见 [CPU协议与验证报告](docs/controlled_pooling_cpu.md)。优化版正式五折未启动，C3/C4未实现。原研究条件见 [冻结协议](experiments/controlled_pooling_adapter_ema/protocol_v1.md)，第一阶段审查见 [配置报告](docs/controlled_pooling_review.md)。此前冻结特征35折全部保留，记录见 [v2_formal_execution.md](docs/v2_formal_execution.md)，不与本实验混表。

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
tests/test_adapted_models_v2.py     # 官方比例数量、梯度与新旧eval一致
tests/test_protocol_v2.py           # epoch>=10资格、早停和外层隔离
experiments/aggregation_comparison/
  src/aggregators.py                # 已完成三基线的原样模型代码
  src/patient_bags.py、metrics.py    # 实际划分、缓存和验证阈值
  src/adapted_models/               # 独立 DGR / ACMIL / mask-safe Nystrom
  src/adapted_models/acmil_official_ratio.py  # 独立ACMIL v2，不改旧入口
  src/protocol_v2.py、inner_cache_v2.py       # 资格协议及inner-only读取
  scripts/train_cv.py               # 基线实际统一入口（存档，不重跑）
  scripts/train_adapted_cv.py        # 新方法的受 smoke gate 约束入口
  scripts/train_adapted_cv_v2.py     # v2默认仅smoke，正式须额外授权和gate
  scripts/verify_v2_runs.py          # 独立v2资格/指标/配对bootstrap核验
  tests/test_verify_v2_runs.py       # 不改变训练gate的6项合成核验测试
  tests/test_protocol.py            # 原真实缓存协议测试源码（存档）
docs/                              # 项目地图、实验状态、许可和审查计划
experiments/controlled_pooling_adapter_ema/
  core.py                          # 复用原Encoder/聚合头/EMA loss的统一封装
  train.py                         # 默认inner-only smoke；正式须新授权与当前gate
  tests/test_controlled.py         # 合成图像/梯度/保存恢复/防泄漏7项测试
  protocol_v1.md                   # 用户确认配置及预先固定的nested协议
experiments/controlled_pooling_cpu/
  loaders.py                       # 保留原逐轮随机轨迹的常驻训练worker
  train.py                         # 作用域内替换loader，复用原统一trainer
  tests/test_cpu_loaders.py         # 四worker、模型/优化器轨迹和异常清理
tools/verify_controlled_pooling.py  # inner-val重放与落盘outer指标独立核验
tests/test_controlled_verification.py # 核验器6项合成测试
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

上述为上一版本记录：旧DGR五折均为 warmup-selected，旧ACMIL五折候选数减1。本轮独立ACMIL v2使用 `min(n_masked_patch,K_valid)` 并最终保留至少一张；DGR fullobj-v2只允许epoch>=10且完成辅助更新的checkpoint，warmup不累计patience。当前22项测试通过；DGR资格smoke10轮、ACMIL smoke2轮均无outer test评估。Dropout只审计、不调整；[参数差异](docs/model_config_diff.md)及[未执行的未来正式命令](docs/review_commands_v2.md)均已记录。

2026-11-15更新：用户随后明确授权两个v2正式五折，前置检查和有效gate核对后已完成。新核验器6项合成测试通过，10个新折的资格/阈值/九指标核验通过，核验阶段outer test模型推理次数0。原始五方法表、v2探索性表、pooled OOF CI和患者配对bootstrap均只保存在本地，没有将真实指标或预测提交GitHub。

## 数据与第三方代码边界

禁止提交医学图像、患者 ID/标签/路径表、临床数据、特征、权重、真实 OOF/逐患者预测、日志、结果 CSV、密钥和个人绝对路径。本仓库不含这些文件，也不含下载的第三方 ZIP/PDF 或完整源码。汇总结果也需另行审查后才能加入。

官方方法的论文、固定版本和许可见 [external_methods.md](docs/external_methods.md)、docs/upstream_source_manifest.json；必要 MIT 声明在 docs/licenses/。未给项目整体擅自附加开源许可证，第三方各自许可仍适用。本轮完成后停下，等待代码审查。
