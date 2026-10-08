> Archived protocol. For the portable entry point, see ../../docs/REPRODUCING.md.

# 集中式与我们的联邦训练：MMST-ViT 配对实验 v1

2026-09-29，用户确认主问题是两种训练方式的预测差距。该协议是作者公开模型上的联邦扩展；不声称复现 CropNet KDD 五州主表，也不使用旧 Ridge 数字填表。

## 固定比较

同一 MMST-ViT 架构分别训练 centralized、FedAvg、hierarchical_async。每个种子只生成一个随机初始化，三组均从它开始，联邦组不加载集中式训练结果。种子固定 10/20/30，200 个数据遍历，三组均处理 222×200 个训练样本。汇报所有种子，最终检查点唯一，没有最佳测试轮次筛选。

主表：RMSE、作者口径 Corr²、有符号 Pearson Corr、标准决定系数（明确标注为补充），以及相同种子下的 RMSE 差值和百分比。正差表示联邦误差增大。图展示 RMSE 和逐县真实/预测散点；预设种子10用于散点，不选最优种子。没有预设非劣效界值，所以差距小不能直接写成统计等价。

## 来源与数据

- 架构：作者 MMST-ViT 固定提交 `615666c8d9fcd704acb662c13065703cbf2eab70`，完整源码保留许可证于 `author_protocol/full_source_615666c8/`。
- 数据：作者 Tiny CropNet 固定 revision `07260d9d23d7093abd89cb5a78f8d82a439c8816`。
- 县表：公开 `input/county_info_2021.csv` 全部222县训练、`county_info_2022.csv` 全部254县评价，覆盖 IL/IA/LA/MS。未沿用公开脚本默认的10县演示筛选。
- 输入：4–9月每月首个选定影像日期、全部县内图块，224×224中心裁剪；每个月前28天九维逐日气象；2017–2021五年月度气象。固定五年窗口照公开配置，训练2021包含该年完整天气，属于回顾性产量预测，不宣称收获前实时预测。
- 保留两输出头及作者 log(production)/log(yield) MSE 目标；主指标由 yield 反变换得到，单位 BU/ACRE。
- 不缩成32×32，不换成图像均值特征，不静默丢县/丢图块/填天气缺失。图块容量超过默认63个时直接报错，必须另立协议扩容；空间位置与气象排列沿用作者数据对齐约定。
- 下载元信息已核验：全矩阵282文件，约57.026 GiB。准备后的逐县张量另占空间。下载可断点续传，并核对上游 LFS SHA256 或 Git blob 校验和。

2022 在本项目历史工作中已查看，必须标为历史基准复测。当前入口把准备、训练和评价分开；训练不读取2022数据，不用其指标早停、调参、选种子。正式泛化确认仍需新的未查看留出。

## 相对作者默认训练的明确差异

采用其完整网络与公开数据格式，使用共同随机初始化，**尚无 MM-SSL 预训练**；固定中心裁剪替代随机增强，沿用其公开 loader 的 float 0–255 强度及固定均值/标准差。没有拟合数据归一化。故不能把所得分数称为作者发表的3.9复现。

三组都用 AdamW、相同基础学习率/余弦日程/梯度裁剪。集中式优化器连续保留状态；联邦各次派发重置本地 Adam 状态，这是本版本明确的算法选择。该对照衡量完整训练方式变化，不把它解释成只由通信引起的误差。没有独立开发集或超参搜索；若效果不佳，保留负结果，另设开发协议后调整。

适配器不改封存源码：修复 PVT 单图块尾批次 `squeeze` 丢批维的问题，加入梯度检查点以容纳完整图块。实际网络共41,628,412参数，两输出头与原结构保留；分类器中未参与 forward_features 的原参数保持不变。

## 分层异步设置

每州按FIPS顺序等分成5个客户端，共20个客户端；一州是一逻辑环。环内客户端从共同模型出发训练一个本地epoch，按样本数聚合。异步环保留派发基点和版本，服务器应用：

`w_next = w_current + server_lr * (n_ring / n_total) * (1 + age)^(-p) * (w_ring - w_dispatch)`

服务器 `server_lr=1`、`p=1`；四环固定逻辑服务时间1/1.2/1.4/1.6。每环恰好完成200个任务，排空在途任务，保证三组实际样本预算一致。事件日志保存版本、陈旧度、派发计算量与接纳样本量。

这是现有通用神经网络增量聚合 `aggregation.fuse` 的接入，**不是 Ridge 专用曲率/梯度缓存求解器的直接复用**。目前未把41M参数网络接入真实轨道事件，也没有重训调度控制器；逻辑时间不解释为秒、时延、带宽或能耗。该阶段先回答预测差距。

## 本地运行

入口 `<repository>\run_mmst_paired.ps1`。首次运行已把 einops 0.6.1、timm 0.5.4、torchvision 0.23.0 装入独立 `.deps-author`，不改旧环境的PyTorch。当前PyTorch为2.8/CUDA12.8；换机器需安装匹配的PyTorch和torchvision。

```powershell
cd <repository>
# 只检查输入，不训练、不产生预测分数
.\run_mmst_paired.ps1 -Stage Plan
# 真正下载完整输入；中断后使用相同Output重跑可续传
.\run_mmst_paired.ps1 -Stage Download -IncludeEvaluation -Fetch -Output results/joint_fusion/mmst_data_download_v1
# 准备训练数据；逐县保留完整时空张量，缺项即报错
.\run_mmst_paired.ps1 -Stage PrepareTrain -Output data/author_tiny_cropnet/prepared_train_v1
# 3方法×3种子，耗时须实际测量；本机执行时保持供电
.\run_mmst_paired.ps1 -Stage Train -Prepared data/author_tiny_cropnet/prepared_train_v1 -Output results/joint_fusion/mmst_train_v1
# 训练与选择冻结后，进行一次历史基准复测并自动出图
.\run_mmst_paired.ps1 -Stage PrepareTest -Output data/author_tiny_cropnet/prepared_test_v1
.\run_mmst_paired.ps1 -Stage Evaluate -Prepared data/author_tiny_cropnet/prepared_test_v1 -Run results/joint_fusion/mmst_train_v1 -Output results/joint_fusion/mmst_evaluation_v1
```

训练和准备目录不覆盖已有结果。每次运行保留协议配置、代码快照、数据哈希、共享初始化、逐方法最终权重与日志。评价输出 `metrics.json`、逐县预测、`comparison.md` 以及PNG/SVG对比图。没有真实评价结果时，绘图程序拒绝生成性能图。

## 连续执行与下载恢复

`python -m research.joint_fusion.run_mmst_pipeline --manifest results/joint_fusion/mmst_paired_setup_20260929/download_manifest.json --out results/joint_fusion/新的流水线目录` 可顺序执行训练数据续传→全量输入校验与准备→三方法三种子训练→补齐评价数据→历史基准评价与出图。优先下载训练输入，避免等待2022文件才开始训练。此顺序调整不改变样本、模型或超参数。

Windows短暂占用进度文件时使用独立临时文件和重试；单次进度写入失败不再中断数据传输。文件内容和上游哈希校验仍然严格。下载目录使用进程级锁，禁止两个任务同时向同一partial文件续写。流水线根status.json区分当前阶段；只有最后阶段completed才代表对比完成。任何输入异常或训练错误都会停止后续阶段并保留失败记录，不静默删县或修改预算。
