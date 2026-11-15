# 聚合替换运行说明

当前阶段已完成开发与inner-only smoke，未运行C0–C2正式五折。后续命令仅供另获授权后执行；不要用旧冻结特征的入口或缓存。

合成测试，不读患者数据、不下载权重：

```sh
python -m unittest discover -s experiments/controlled_pooling_adapter_ema/tests -v
```

在仓库根目录设置本地变量：`IMAGE_INDEX`（原图路径与SHA256索引）、`IMAGENET_WEIGHTS`、`REFERENCE_ARGS`、`NEW_SMOKE_ROOT`；目录必须位于Git工作树之外，输出fold不得已存在。索引包含patient_id、label、fold、resolved_path、sha256，其实际内容仅留本地。

```sh
python experiments/controlled_pooling_adapter_ema/train.py --group all --mode smoke --fold 0 --index "$IMAGE_INDEX" --weights "$IMAGENET_WEIGHTS" --reference "$REFERENCE_ARGS" --run-root "$NEW_SMOKE_ROOT"
```

以下是**尚未执行**的未来15折命令。另行授权且审查确认后，设置全新 `NEW_FORMAL_ROOT`、当前 `PASSED_SMOKE_ROOT`，保留正式workers=4。Windows PowerShell使用 `$env:NAME` 或本地 `$NAME` 变量；命令中的`$NAME`对应后者。默认自动选择CUDA（如可用），否则CPU；GPU环境尚未验收，应先重做同一smoke。不要凭CPU微型smoke外推正式耗时/显存。

```sh
python experiments/controlled_pooling_adapter_ema/train.py --group C0 --mode formal --fold all --formal-authorized --gate-root "$PASSED_SMOKE_ROOT" --index "$IMAGE_INDEX" --weights "$IMAGENET_WEIGHTS" --reference "$REFERENCE_ARGS" --run-root "$NEW_FORMAL_ROOT" --workers 4
python experiments/controlled_pooling_adapter_ema/train.py --group C1 --mode formal --fold all --formal-authorized --gate-root "$PASSED_SMOKE_ROOT" --index "$IMAGE_INDEX" --weights "$IMAGENET_WEIGHTS" --reference "$REFERENCE_ARGS" --run-root "$NEW_FORMAL_ROOT" --workers 4
python experiments/controlled_pooling_adapter_ema/train.py --group C2 --mode formal --fold all --formal-authorized --gate-root "$PASSED_SMOKE_ROOT" --index "$IMAGE_INDEX" --weights "$IMAGENET_WEIGHTS" --reference "$REFERENCE_ARGS" --run-root "$NEW_FORMAL_ROOT" --workers 4
```

每组输出在 `NEW_FORMAL_ROOT/C0|C1|C2/fold_0..4/`。全部真实split、val/test预测、checkpoint和训练日志只留本地。任何源码、参考配置、索引、权重变化都会使旧gate失效；不能手改gate绕过。正式性能表和独立指标核验属于下一阶段，不是当前已完成结果。
