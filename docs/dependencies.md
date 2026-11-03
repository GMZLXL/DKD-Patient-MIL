# 依赖和验证范围

requirements.txt 是本机直接依赖的已安装版本记录，不是声称从零重建并验证全部训练环境。已验证 Python 3.10.11 / CPU 的语法、导入、CLI --help 和合成特征检查；没有安装新环境、下载权重或执行真实训练。

torchvision 提供 EfficientNet-B0；原模型构造函数可能请求下载 ImageNet 权重，故轻量测试只实例化患者聚合头与 ConvAdapter，不构造图像编码器。训练使用者须自行提供或配置权重缓存，并核对来源。

RETFound 入口在 build_retfound_encoder 内延迟导入使用者指定源码中的 models_vit、util.pos_embed。因此顶层 import 和 --help 通过不能证明 RETFound 编码器运行依赖齐全。本仓库不包含 RETFound 源码/权重，不宣称验证过实际模型加载；相关 timm 等依赖需按使用者固定的 RETFound 版本安装。对 checkpoint 的 weights_only=False 是保留的历史行为，只应载入使用者信任的本地文件。

历史预处理工具依赖 OpenCV；元数据与患者表工具依赖 pandas。所有数据读写发生在使用者显式运行入口时；本阶段未运行这些 main 函数。
