# 代码审查整改报告

审查基准：`GMZLXL/DKD-Patient-MIL@aa6137351f5ace7e67b943f3de8587526eb3025f`，分支 chore/research-code-audit。本轮**没有启动新的正式五折实验**，完成修订和测试后停止等待下一次审查。

## 实现与历史兼容

1. 新增 ACMIL-GA-adapted-v2-official-mask。旧acmil.py和旧入口原样保留。候选数由 `min(10,K-1)` 改为 `min(10,K)`，最终丢弃上限仍为K-1；K=1..9默认数量为旧 `[0,0,1,1,2,3,3,4,4]`、新 `[0,1,1,2,3,3,4,4,5]`。架构、初始化布局、5分支、CE及attention diversity权重均不改。
2. 新增 DGR-MIL-adapted-fullobj-v2 版本化协议/入口。模型和损失保持原样，辅助目标从epoch10起启用；checkpoint只有epoch>=10且有实际辅助optimizer更新才合格。warmup不保存最佳、不累计patience。合格后仍按内层val AUC优先、CE次之，预算100、patience20。
3. 新增Dropout参数审计及非运行配置模板，详见 model_config_diff.md。上游训练入口 patch=.1、node=.4；现有适配无投影Dropout，两个self-attention输出Dropout=.1，cross probabilities Dropout=.1。隐藏256而非入口512，Dropout类型/位置差异明确记录。本次未改变旧网络或调参。
4. 新增smoke专用缓存读取：文件级SHA校验后memory map，仅为inner开发患者构造标签/特征tensor；外层bag只保留fold，新split不读取外层标签或特征。旧缓存和原分层结果不变。

## 实际验证

- **22项默认公开单元测试通过**（此前11项保留）。新增覆盖精确掩蔽数、固定seed复现、mask_drop=1、K=1..9的有限输出/梯度、padding零梯度、不影响预测、eval顺序无关、新旧同权重eval相同；真实DGR辅助项对全局表征有非零梯度。
- 资格回归覆盖warmup从不保存/从不累计patience、首个合格epoch建立基准、AUC/CE tie-break、无辅助更新证据时报错、错误best_epoch报错、每个保存模型都在epoch>=10且有辅助更新。
- 外层标签翻转或禁止访问不改变checkpoint、参数或阈值；完整新split与旧分层结果逐fold一致。smoke loader用人工非法外层标签/NaN向量测试，证明不解析这些标签、不构造这些向量；不是放宽正式缓存验证规则。
- 旧DGR入口的人工inner-only回归仍能选warmup，旧ACMIL builder仍指向旧类。原27个Python源码前后SHA256一致。
- 当前最终源码版本的真实fold0 smoke：inner train抽8人、inner val抽4人。DGR运行10轮，best_epoch=10、auxiliary_active_at_best_checkpoint=true、已完成1次辅助optimizer更新；ACMIL运行2轮，best_epoch=2。两者checkpoint保存/加载一致，**outer_test_evaluations=0**，无外层预测文件。
- 先完成过普通inner smoke，随后新增更严格的缓存隔离，并在独立新目录重新完成上述最终smoke；两次输出均保留，不覆盖旧smoke。原测试中使用人工线性模型的synthetic test fold不涉及真实外层患者。
- `python tools/check_release.py`、实际暂存内容检查及git diff检查通过。只提交代码、脱敏配置说明和文档，真实运行配置/证据CSV/日志/模型均留本地。

## 历史保护与解释边界

开始时在仓库外保存 baseline_fingerprint_before.json；最终核对 **234个历史产物与缓存文件、27个历史Python源码SHA256全部一致**。既有25折和每方法205条OOF不重跑、不覆盖，也未加载旧模型做外层预测。正式外层测试产物文件数仍为25。

新版协议是在已观察旧结果后提出的**事后探索性/敏感性分析**，不是事前注册的确认性比较。修订/测试通过不等于科学复现；本阶段没有新AUC/性能结论，未将真实性能数字写入公开报告。旧DGR仍标注warmup-selected，两个v2正式五折均未运行。

精确后续CLI、预计独立输出目录结构和额外授权/gate前置条件见 review_commands_v2.md。实际本地路径命令另存于仓库外，不上传个人路径。当前不执行这些正式命令。
