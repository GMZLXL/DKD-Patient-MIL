# CPU执行优化：独立后端与验证协议

用户选择停止当前正式运行、保留产物、先验证CPU优化。旧C0 fold0完成37个epoch后停止，best checkpoint可读取（epoch4），没有完成的outer fold；C1/C2尚未开始。该未完成运行不是正式五折结果，不与历史35折冻结特征结果混用。原目录、checkpoint、训练记录及已审查训练源码均保留。

新增独立 `experiments/controlled_pooling_cpu/`，直接调用原 `controlled_pooling_adapter_ema.train.run_fold`。没有重新实现Encoder、聚合头、EMA loss、epoch、scheduler、checkpoint选择、阈值或外层评估。使用限于已验证的PyTorch2.7.0 CPU环境；其他环境报错，需要重新验证。

## 执行差异

| 项目 | 原执行 | CPU后端 |
|---|---|---|
| 训练worker | 每epoch重新创建4个 | 4个常驻worker，每折结束显式释放 |
| 验证/最终测试worker | 4个 | 主进程加载，0个worker |
| 模型CPU线程 | 2 | 2，不调整 |
| 随机数 | 新迭代器生成base_seed并初始化worker | 显式模拟相同生成器消耗，每轮按同一base_seed+worker_id重新设置Torch/Python/NumPy |
| 模型与训练条件 | 原冻结协议 | 完全沿用 |

普通的 `persistent_workers=True` 会改变原共享generator的消耗顺序和worker增强轨迹。因此新增SeededSampler/SeededDataset封装：首轮预览原base_seed但不提前消耗；以后每轮显式消耗原迭代器本应消耗的base_seed，再由原RandomSampler抽样；首个样本触发每个worker按原种子重置。图像仍经过原Dataset的随机增强，再完整走可微Conv-Adapter路径。没有训练特征缓存。

验证Dataset没有随机增强；主进程加载保留同一batch/顺序/张量及parent RNG。训练池常驻时不再另驻留4个验证worker，控制内存开销。优化不减少验证频次或训练预算，不关闭确定性算法。

新入口使用作用域内依赖注入，仅替换loader和执行指纹/元数据；退出或异常时恢复baseline全局函数，finally释放worker。配置额外记录train/val/test worker、重置规则、模型线程及版本。原CLI/源码没有被重定向。

## 验收规则

1. 四worker连续三轮：样本顺序、Torch/Python/NumPy随机值、parent RNG、AdamW模型/优化器轨迹精确一致，worker PID保持且关闭后退出。
2. 原Dataset合成图像K=1–9连续两轮随机增强张量一致；验证4-worker和0-worker张量及parent RNG一致。
3. 原7项模型/梯度/K/EMA/保存恢复/数据隔离测试继续通过。
4. C0/C1/C2各做真实384图像fold0 inner-only smoke，train4名/val2名、各2epoch，4个训练worker、0个验证worker，不进行outer推理。
5. 用fold0全部131名inner train（451图）及33名inner val（109图）进行顺序配对测速，每种执行3epoch，模型线程2、所有科学参数不变。逐epoch比较输入张量与顺序SHA、模型/criterion/optimizer/scheduler、主进程和loader RNG、两种val概率及训练记录；baseline还须与原中断运行前3轮记录一致。
6. 测速只比较时间、内存及等价性，不选择测试指标；不读取outer图像做模型推理，不构成正式医学结果。单次配对和共享内存重复计数限制需要报告。

验证日志、真实Smoke预测/权重及全部测速产物仅留本地。耗时数字和通过状态以本次完成后的执行报告为准；不能以微型smoke外推正式训练时间。

## 独立核验器

`tools/verify_controlled_pooling.py` 与 `tests/test_controlled_verification.py` 是正式训练启动阶段新增的只读核验代码，本次一并纳入源码清单。它复现inner选模/阈值、重放inner val并检查EMA只读，九项outer指标从落盘预测独立复算，合并OOF使用各患者所属折阈值；没有outer模型重放路径。当前没有15个完整折，不运行该正式核验程序、不生成正式比较表。

## 新门禁及下一阶段

CPU门禁额外绑定本后端及其合成测试SHA256；旧门禁不能替代。CLI默认smoke，正式需重新明确授权和当前CPU gate。当前任务仅到验证和报告，不启动优化版五折。

合成测试可单独运行（不下载权重、不读真实数据）：

```sh
python -m unittest discover -s experiments/controlled_pooling_cpu/tests -v
```

GPU资源未新增；384图像、batch4、seed42、100epoch/patience50、原W-BCE+EMA-PCAR及nested划分和raw评估口径均按 [protocol_v1.md](../experiments/controlled_pooling_adapter_ema/protocol_v1.md)。原入口仍只有best checkpoint，没有完整last/RNG续训；不得声称从第37轮无损续跑。未来采用新的正式输出目录、从统一ImageNet/seed42初始状态开始，不载入中断任务的best作为初始化。

未来命令（本阶段未执行），变量路径由本地提供且输出目录在Git树之外：

```sh
python experiments/controlled_pooling_cpu/train.py --group all --mode smoke --fold 0 --index "$IMAGE_INDEX" --weights "$IMAGENET_WEIGHTS" --reference "$REFERENCE_ARGS" --run-root "$NEW_CPU_SMOKE_ROOT" --workers 4 --threads 2
python experiments/controlled_pooling_cpu/train.py --group all --mode formal --fold all --formal-authorized --gate-root "$PASSED_CPU_SMOKE_ROOT" --index "$IMAGE_INDEX" --weights "$IMAGENET_WEIGHTS" --reference "$REFERENCE_ARGS" --run-root "$NEW_CPU_FORMAL_ROOT" --workers 4 --threads 2
```

## 本轮实际验证结果（2026-11-15）

46项合成/回归测试全部通过（CPU后端5、原模型协议7、顶层28、历史核验器6）。C0/C1/C2真实384图像fold0 smoke各2epoch通过，训练4-worker、验证0-worker，模型+EMA恢复精确一致，outer test推理0。

| C0内层3轮配对指标 | 原加载方式 | CPU后端 |
|---|---:|---:|
| 3轮训练+验证合计秒 | 290.246 | 186.311 |
| 第2–3轮平均每轮秒 | 96.751 | 54.748 |
| 数据迭代等待合计秒 | 127.465 | 21.296 |
| 主进程+子进程工作集峰值MiB | 3145.2 | 3080.8 |


三轮合计提速 1.558 倍；第2–3轮平均提速 1.767 倍。仅C0完整inner train/val负载的一次顺序配对测试，包含输入SHA审计开销；工作集求和会重复计入共享页，不能当成独占物理内存。其他组完整正式耗时尚未测量，不承诺未来同等倍率或完成时间。

两种执行逐epoch的输入、模型、EMA/criterion、optimizer、scheduler、parent/loader RNG及raw/calibrated val概率全部精确一致。原执行的前3轮记录还与已中断原任务前3轮精确一致。指标没有用于挑选优化设置，未进行outer模型推理。

当前没有新的正式折或正式比较表；原37轮未完成任务保留，旧35折及已登记历史产物不变。新版只通过测试/Smoke/配对执行验证，停在此处等待审查与新正式授权。

以上为CPU验证阶段的历史状态。用户随后明确授权启动优化版正式15折。启动前只修正独立核验器：类别权重按训练时实际保存的float32表示比较，验证回放补全原Dataset必需的fold字段；新增两项合成回归，核验器8项测试及三组已保存inner-only smoke的实际回放通过。模型、损失、CPU后端和其门禁绑定的11个训练/测试源文件逐字节不变。正式执行使用独立新目录，从统一初始化开始，不恢复或覆盖原37轮任务；运行状态和真实产物仍只留本地。
