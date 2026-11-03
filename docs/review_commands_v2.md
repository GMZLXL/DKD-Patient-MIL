# v2 命令与前置条件

本阶段**没有执行正式五折**。以下正式命令只为下一次审查提供可复核接口，必须另获用户明确授权后才可运行；`--formal-authorized` 是操作者对该授权的显式确认，smoke通过不代表授权。

## 本阶段的检查与inner-only smoke

```sh
python -m unittest discover -s tests -v
python tools/check_release.py
python experiments/aggregation_comparison/scripts/train_adapted_cv_v2.py --method dgr_fullobj --mode smoke --fold 0 --smoke-epochs 10 --cache "<LOCAL_FROZEN_B0_CACHE>" --run-root "<LOCAL_NEW_V2_SMOKE_ROOT>"
python experiments/aggregation_comparison/scripts/train_adapted_cv_v2.py --method acmil_official_ratio --mode smoke --fold 0 --smoke-epochs 2 --cache "<LOCAL_FROZEN_B0_CACHE>" --run-root "<LOCAL_NEW_V2_SMOKE_ROOT>"
```

smoke专用loader仅在文件字节层核验缓存SHA，使用memory map，**不会构造外层患者的标签或特征tensor**；外层bag只保留fold以维持原分层划分的患者集合。新split不读取外层标签。只在inner train抽8人、inner val抽4人；不调用outer test预测。DGR的第10轮确有辅助目标optimizer更新，之后才保存合格checkpoint。每次运行目录已存在就拒绝覆盖。

## 未来另获授权后才可执行的正式命令（本次未执行）

```sh
python experiments/aggregation_comparison/scripts/train_adapted_cv_v2.py --method dgr_fullobj --mode formal --fold all --formal-authorized --cache "<LOCAL_FROZEN_B0_CACHE>" --run-root "<LOCAL_NEW_V2_FORMAL_SENSITIVITY_ROOT>" --gate "<LOCAL_NEW_V2_SMOKE_ROOT>/dgr_fullobj/fold_0/smoke_passed.json"
python experiments/aggregation_comparison/scripts/train_adapted_cv_v2.py --method acmil_official_ratio --mode formal --fold all --formal-authorized --cache "<LOCAL_FROZEN_B0_CACHE>" --run-root "<LOCAL_NEW_V2_FORMAL_SENSITIVITY_ROOT>" --gate "<LOCAL_NEW_V2_SMOKE_ROOT>/acmil_official_ratio/fold_0/smoke_passed.json"
```

正式预算在版本化CLI中锁定 seed42、100epoch、patience20、AdamW lr=.001、weight_decay=.0001、patient batch16。DGR最佳checkpoint资格从epoch10开始，不改变目标、中心、Dropout、特征或原fold。阈值仍只由内层val F1选取。

预计输出为新根目录下 `dgr_fullobj/fold_0..4/`、`acmil_official_ratio/fold_0..4/`，与旧method名及已完成结果隔离。新增checkpoint_selection.csv逐轮记录内层证据。只有正式执行才会在选模和阈值固定后评估一次outer test；smoke路径没有测试预测文件。

前置条件：下一次代码审查通过、用户另行授权、同一缓存完整性验证通过、对应v2 gate的源码/测试摘要与缓存摘要/seed匹配、DGR gate确实验证epoch>=10且辅助更新次数>0、新输出目录不存在。任何源码/测试改动会使旧gate失效；不得自动沿用旧版2epoch gate，也不得指向历史输出根目录。

这些是事后敏感性实验，不替换既有25折，不当作预注册确认性比较。所有真实配置、模型、日志、证据CSV和预测继续只留本地。
