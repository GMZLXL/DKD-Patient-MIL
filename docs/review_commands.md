# 本次入口与复核命令

以下路径均为本地占位符，不会由脚本自动猜测。真实缓存和输出不得加入 Git。

```sh
python -m unittest discover -s tests -v
python tools/check_release.py

# 必须先完成每个方法的单折 smoke；正式命令的 gate 必须与源码一致。
python experiments/aggregation_comparison/scripts/train_adapted_cv.py --method dgr_mil --cache <LOCAL_FROZEN_B0_CACHE> --run-root <LOCAL_NEW_SMOKE_DIRECTORY> --fold 0 --smoke
python experiments/aggregation_comparison/scripts/train_adapted_cv.py --method acmil --cache <LOCAL_FROZEN_B0_CACHE> --run-root <LOCAL_NEW_SMOKE_DIRECTORY> --fold 0 --smoke

python experiments/aggregation_comparison/scripts/train_adapted_cv.py --method dgr_mil --cache <LOCAL_FROZEN_B0_CACHE> --run-root <LOCAL_NEW_FORMAL_DIRECTORY> --gate <LOCAL_DGR_SMOKE_PASSED_JSON> --fold all --seed 42
python experiments/aggregation_comparison/scripts/train_adapted_cv.py --method acmil --cache <LOCAL_FROZEN_B0_CACHE> --run-root <LOCAL_NEW_FORMAL_DIRECTORY> --gate <LOCAL_ACMIL_SMOKE_PASSED_JSON> --fold all --seed 42
```

三种已完成基线的原统一入口 `scripts/train_cv.py`、原 `scripts/smoke_test.py` 及真实缓存协议测试仅为实际执行代码存档，本轮未调用它们重训。若以后经审查决定重复实验，需使用全新输出目录，且显式准备对应的本地缓存/报告；默认目录不是公共数据下载服务。

公开仓库顶层测试全部使用合成数据。`experiments/aggregation_comparison/tests/test_protocol.py` 是原实际用过的真实缓存版本，保留逐字节源码以供核对，不属于无需数据的默认 unittest discover 范围。
