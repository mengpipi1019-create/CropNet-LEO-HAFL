# 本地复现

## 1. 环境

推荐 Python 3.11 或 3.12。原训练环境为 Windows、PyTorch 2.8.0 + CUDA 12.8、torchvision 0.23.0、RTX 5070 Laptop 8 GB。模型训练入口要求 CUDA；图表与元数据核验可在 CPU 上运行。

在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

显卡与驱动需支持所选 PyTorch 版本。其他平台可安装匹配版本并使用下方 `python -m ...` 命令；Windows 启动脚本已提供，其他系统未做完整 GPU 长跑验证。

## 2. 下载和准备

```powershell
python -m research.joint_fusion.prepare_mmst_compare --plan
python -m research.joint_fusion.prepare_mmst_compare
```

`--plan` 只核对固定数据版本和县列表，不下载、不训练。完整准备会下载 Hugging Face 的固定版本，逐文件校验 SHA-256/Git blob，并生成训练和评估缓存。原始文件约 57 GiB，缓存、9 组训练状态及恢复文件还需额外空间；建议新工作盘预留至少 350 GiB，训练前还会执行实际容量检查。

准备命令支持重复执行。Ctrl+C 可停止准备，重新执行继续；中途损坏或不匹配的文件会明确报错，不会静默跳过。已有原始数据可放入本仓库 `data/author_tiny_cropnet/raw/`，保持固定下载清单中的相对路径及 `.source.json` 校验侧文件。

新准备目录为 `results/joint_fusion/portable_inputs/`，不依赖原机器的校准实验目录。不需要先做旧的短诊断；历史摘要随仓库保存仅供追溯。

## 3. 正式比较、状态和安全暂停

```powershell
.\start_mmst_compare.cmd
.\run_mmst_compare.ps1 -Action Status
.\pause_mmst_compare.cmd
```

若没有仓库内 `.venv`，显式选择已有 Python：

```powershell
.\run_mmst_compare.ps1 -Action Run -Python 'C:\path\to\python.exe'
```

跨平台对应命令：

```text
python -m research.joint_fusion.run_mmst_compare --action check
python -m research.joint_fusion.supervise_mmst_compare --out results/joint_fusion/mmst_compare_balanced_20260930_01
python -m research.joint_fusion.run_mmst_compare --action status
python -m research.joint_fusion.run_mmst_compare --action pause
```

等待显示 `PAUSED` 后再关训练窗口。相同启动命令恢复；训练期间不要修改源码、配置、已准备数据或运行输出。恢复依靠固定源码和文件校验，永久错误会停止，暂时错误最多重启 3 次。原设备此次训练约 47.7 小时；新机器耗时取决于 GPU、磁盘、散热和数据准备情况。

## 4. 配对协议

训练年 2021 共 222 县；2022 历史评估共 254 县；种子 10、20、30，每种方法每种子 120 个已接收样本轮次等价预算。三方法均处理完整训练样本预算，但优化器步数和聚合次数不同，不能称为等墙钟时间或等 FLOPs 比较。

集中式批大小 10；联邦每客户端每轮完整本地数据 1 次更新；异步四环采用固定逻辑延迟、最大领先 2，平衡系数 0.9066011235955057、陈旧度指数 0。归一化只用训练年拟合。

集中式 / FedAvg / 异步使用相同结构和配对随机初始参数，无 SSL 预训练。全部训练完成后才计算历史评估分数；不按测试分数调参或选择检查点。作者 R² 定义为 Corr²，同时保留标准决定系数 `1 - SSE/SST`，避免混淆。

## 5. 数据留存与限制

新运行保留源码快照、配置、数据校验、归一化、客户端划分、优化器状态、滚动恢复检查点、最终模型、逐县最终预测、训练事件日志及训练探针曲线。当前入口沿用已完成实验协议，只做最终测试；不会自动生成逐轮测试预测，也不会保存全部 120 轮完整模型。若研究需要其他留存策略，应建立新的配置和输出目录。

仓库自带结果在 `examples/`，新运行写入被 Git 忽略的 `results/`。两者互不覆盖。
