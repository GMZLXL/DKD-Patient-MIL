# 阶段 0 检查记录

- 原项目不是 Git 仓库，没有可审查的既有 Git 历史或暂存内容；创建独立的新代码仓库，未导入旧历史。不存在从旧 Git 历史携带患者数据的问题。
- 导出使用7个脚本白名单。保留目录/文件名；除 RETFound 的2个个人绝对路径默认值变为必填参数外，不改算法或训练协议。source_manifest.json 记录原文件与副本摘要。原文件摘要在本地再次核对。
- 10 个 Python 文件通过语法解析；scripts_qc 中8个模块（含包初始化）导入通过。没有实例化预训练 B0/RETFound 编码器或下载权重。
- unittest 的3项 CPU 合成测试通过：Mean/Gated/PHPR 的 K=1/2/9 有限数与顺序不变、单患者与变长联合处理一致、零初始化 ConvAdapter 为恒等变换。未做真实数据测试或训练。
- 4个主要训练入口的 --help 检查通过：冻结 Mean、PHPR、EMA-PCAR、RETFound。--help 通过只说明 CLI 可加载，不证明数据、权重或 RETFound 运行依赖齐全。
- .gitignore 采用根目录允许列表并排除数据、特征、权重、医学图像、患者/结果 CSV、notebook、日志、输出目录、密钥和本地配置。暂存采用明确文件/目录列表，不执行 git add .。
- 提交前检查暂存文件名、实际暂存内容、语法、常见凭据模式和个人路径。新仓库只包含源代码与文档；不包含数据、缓存、真实预测、指标表、第三方 ZIP/PDF 或完整项目。
- 许可：Gated-ABMIL、DGR-MIL、ACMIL 的原 LICENSE 已确认 MIT；AttriMIL 仓库标注 Apache-2.0，但完整 LICENSE 尚需实际移植前确认；TDA 当前仅 available soon。未引入这些官方项目源码。
- 已知未验证：真实医学数据上的运行、GPU/CUDA、模型权重加载、RETFound 延迟导入依赖、完整五折/嵌套训练协议。历史留出折选模行为保持原样，见 project_map.md。

阶段 0 的范围到代码发布与审查为止。发布完成后不自动开始下一阶段。
