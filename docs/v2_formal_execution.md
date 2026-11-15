# v2正式执行与独立核验完成状态（2026-11-15）

训练审查基准：ed3113384ecee3f966f70881886c2f40fba73b76，分支 chore/research-code-audit。用户在本轮明确授权前置检查通过后运行两个现有v2各五折。本次没有修改训练入口、模型、超参数、冻结特征或患者划分。

## 前置检查与执行

本地和远端HEAD一致、工作区干净。原22项合成测试、源码安全检查和CLI检查通过。703×1280冻结ImageNet EfficientNet-B0缓存的文件SHA、索引、205名患者及K=1..9核对通过。两个已有inner-only smoke gate的源码/测试/缓存摘要、seed、资格字段完全匹配，因此直接复用，没有重复smoke训练。新输出根目录起初不存在，且与旧25折目录隔离。

直接调用 train_adapted_cv_v2.py，使用 --mode formal --formal-authorized --fold all，方法仅为 dgr_fullobj 与 acmil_official_ratio。固定seed42、100epoch上限、patience20、AdamW lr=.001、weight_decay=.0001、patient batch16；每个CPU进程2线程。所有旧33个Python文件内容保持不变。

## 完成事实

| 方法/范围 | 状态 |
|---|---|
| 原始5方法 | 原25折保留，未重训、未覆盖 |
| DGR-MIL-adapted-fullobj-v2 | 新5折完成，全部best_epoch>=10且有辅助optimizer更新 |
| ACMIL-GA-adapted-v2-official-mask | 新5折完成，使用新版候选比例与原GA目标 |
| 新版OOF | 每方法205条、合计410条，按方法版本唯一 |
| 总正式折数 | 原25 + 新10 = 35，不能混称25折 |
| 失败/补跑 | 无失败折、无补跑或重试；两个正式进程均退出0 |

## 独立核验

新增 scripts/verify_v2_runs.py，未使用旧25折核验器冒充v2核验。逐折核对10项必要文件、与原Mean完全相同的train/val/test患者ID、标签、合格epoch内的AUC/CE最优checkpoint、warmup不选模/不计patience、实际辅助更新次数、内层阈值，以及保存测试预测的全部九指标。

核验只重放inner validation，以检查保存的checkpoint与验证概率、AUC、CE和阈值一致；从不对outer test重新推理。正式每折测试一次的核对基于固定训练源码路径及落盘报告，核验阶段新增outer test model evaluations=0，不等同硬件级执行跟踪。

原22项测试保持不变；新增核验器的6项合成测试单独在实验tests目录执行，覆盖拒绝warmup checkpoint、缺辅助证据、错误patience、非最佳合格模型、独立指标/阈值及配对患者bootstrap一致性。新增核验代码/测试不改变现有模型训练gate指纹。两套测试均通过；发布安全检查通过。

## 统计与隐私

原始五方法与v2探索性结果分表，三基线只复用原记录。每表九指标提供五折均值及样本SD(ddof=1)，PR-AUC定义为average precision。205条拼接OOF的pooled九指标另表报告，配对重采样以患者为单位，2000次、seed42、无单类别样本跳过；各方法使用相同索引。CI以固定OOF和各折阈值为条件，不含重训/划分不确定性。

234个历史产物/缓存文件、33个历史Python源码的SHA前后完全一致。真实预测、患者ID/标签、模型、缓存、日志、配置、所有性能CSV和运行报告均只留本地；本仓库仅发布核验程序、合成测试和脱敏状态。本文件不公布真实性能数字。

## 科研边界及后续

两种v2是在看到旧结果后制定的探索性敏感性分析，不能替换旧DGR/ACMIL、冒充独立验证或预注册确认性结论。样本仅205名患者、单seed、固定单候选预算；各头与专用目标不同，属于患者级聚合方法比较，不是只替换数学池化算子的完全同参数消融。没有为追求测试表现调整任何设计，也没有扩展其他方法。

本轮执行与提交完成后停止，等待对真实本地结果及核验代码的审查；不自动开展额外实验。

## 复核接口

```sh
python -m unittest discover -s tests -v
python -m unittest discover -s experiments/aggregation_comparison/tests -p test_verify_v2_runs.py -v
python tools/check_release.py
python experiments/aggregation_comparison/scripts/verify_v2_runs.py --preflight "<LOCAL_PRIVATE_PREFLIGHT_JSON>"
```

preflight清单包含本地实际路径、训练源码与缓存摘要、原历史指纹和全新报告根目录，只在本地生成并保留；不要上传。核验器拒绝覆盖已有报告目录，bootstrap默认2000次。训练参数以实际保存config_resolved.yaml为准，文档YAML不驱动训练。
