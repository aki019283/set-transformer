# DCN-V2 + Set Transformer 多因子选股

这是 dcn_set_s42_v1 历史实验的中文注释版。模型接收一只股票在一个截面上的 103 个因子，输出一个连续分数。分数用于在同月股票池中排序，不能直接读作预期收益率。历史实验采用 seed 42。与同一输入面板的 FT-Transformer 比较，修正估值后的标准年化夏普分别为 **1.364730 和 1.476660**。

## 模型如何产生一个分数

输入形状为 [股票数, 103]，因子来自 Ashare_new 58 列和 rcykc_multi 45 列。以一只股票为例，模型同时计算三部分表示。

1. **显式交叉**。令 h₀=x，每层计算 hₗ₊₁=hₗ+x⊙(UₗVₗhₗ+bₗ)。两层各自使用 103→16→103 的分解。乘号表示同位置元素相乘。这样某个因子的影响可以随其他因子的取值变化，最后仍得到 103 个数。
2. **深层分支**。原始 103 维输入另经 64、32 两层全连接，使用 ReLU，第一层激活后使用 0.1 dropout。该分支学习非线性组合，输出 32 个数。
3. **因子族集合分支**。按列名分成 calc、basic、defense、Alpha101、Canlander、gf125、GTJA、TALib131 八族。每族有自己的线性映射，把不同数量的列转换成一个 16 维向量。四个可学习的诱导向量先通过双头注意力读取八个因子族，再由八个族向量读取这四个摘要（ISAB）。随后用一个可学习查询做注意力池化（PMA），得到 16 个数。股票之间的依赖来自上游截面排名；注意力本身只作用于同一只股票的不同因子族。

最后拼接 103+32+16=151 维，线性层输出一个分数。族内每个输入位置对应固定因子，因此训练与推断必须沿用同一列序。对应实现见 [model.py](model.py) 和 [set_block.py](set_block.py)。

交叉层参考 Wang 等的 [DCN V2](https://arxiv.org/abs/2008.13535)，集合分支参考 Lee 等的 [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html)。按金融因子族生成 token、三路拼接和选股目标是本实验设计；本实验没有使用 DCN-Mix 专家门控。

## 数据、标签和训练

- 每月分别给每列因子排百分位名次，并列取平均，再映射到 [-1,1] 范围内。先对原始数值排名，再转 float32，避免极大有限值提前转换时溢出。保留缺失位置到模型输入阶段，按实验配置填 0。
- 日期处理针对该次存档面板。程序根据来源记录调整月份，并按交易日历记录源月份、持有月、进出场日期。标签用本月首个交易日收盘至次月首个交易日收盘的价格重建。运行其他批次前应复核供应方的月份定义，避免对已标成持有月的输入再次移月。
- 使用原配置的滚动窗口（训练48个月、验证12个月、测试1个月），检查窗口间隔及收益标签成熟时间。模型以收益截面秩为目标，采用 MSE、AdamW 和验证集早停。首窗完整训练，后续窗口沿用前窗权重。微批上限为256行，梯度按微批样本占比累加。
- 完整预测覆盖2021年1月至2026年8月的68个持有月。下表仅使用两种模型均有完整配对回测的22个收益月。两者不能混作同一个评估区间。

## 历史比较

以下数字来自云端 comparison_22m/summary.csv，收益区间为2024年2月至2025年11月。CAGR 和回撤均按百分比展示。

|模型|估值与状态规则|月数|标准年化夏普|CAGR|最大回撤幅度|
|---|---|---:|---:|---:|---:|
|DCN-V2 + Set|repriced_net|22|1.364730|38.9856%|15.3424%|
|FT|repriced_net|22|1.476660|44.0879%|14.5066%|
|DCN-V2 + Set|historical_m0_next_flags|22|1.636410|38.2063%|13.5586%|
|FT|historical_m0_next_flags|22|1.738072|43.3210%|12.5279%|

以历史 M0 股票池、最高十分组 Group10 和简化费用为基础。repriced_net 使用同日期交易状态，固定选中持仓，按预定月初收盘价重估。historical_m0_next_flags 保留旧交易状态时点问题，仅用于对账。月度标准夏普为 sqrt(12) × mean(r−rf) / std(r−rf, ddof=1)，rf=(1.02)^(1/12)−1，r 为月净收益。混合模型在修正口径下夏普低0.111930。现有比较只覆盖一个种子和反复查看过的历史区间，因子披露滞后亦未逐项核实。

## 如何运行

先在本目录做结构检查，只需 Python 和 PyTorch。测试使用随机张量，检验输出形状、不同股票之间的隔离、集合池化顺序以及各分支反向梯度。

~~~powershell
python -B .\check.py
~~~

训练使用 Windows 云端已有的 mlp_factor 环境，含 PyTorch、NumPy、pandas、PyArrow、PyYAML 和项目依赖。-Project 应指向带 src、config 和 data 的项目根目录。配置为 config/mlp_warm/old(58)+history-rcykc-multi(45).yaml，因子列清单见 reference/features.json。行情、交易日历、交易状态、股票池、指数行情和因子面板须由运行者在项目数据目录提供。

~~~powershell
.\run.ps1 -Project 'F:\aki\mlp_factor' -Mode Smoke
~~~

Smoke 只跑两个短窗口。完整比较前，从原实验包的 reference 目录将 ft_transformer_s42.parquet、schedule.csv、monthly.csv 放入本目录的 reference 子目录，再执行：

~~~powershell
.\run.ps1 -Project 'F:\aki\mlp_factor' -Mode Full
~~~

公开版默认写入项目下 data/experiments/dcn_set_s42_public_v1。训练入口会保存源码和输入文件哈希；中文注释改变源码哈希，因此使用独立输出目录。重复执行同一命令时，程序核验已提交窗口，再继续后续窗口。

## 到哪里看结果

|相对 full 的位置|用途|
|---|---|
|windows/dcn_set_s42/|逐窗权重、预测和提交校验记录|
|factors/dcn_set_s42.parquet|回测前的预测分数，TradingDate 为该历史实验的信号标签|
|report/holding_month_schedule.csv|把信号标签映射到实际持有月份、入场与退出日|
|report/monthly_ic.csv|逐月 Pearson IC、Spearman Rank IC 和可用标签数量|
|report/comparison_22m/summary.csv|同口径模型比较，另有各模型逐月收益|
|report/status.json|训练、预测及历史比较完成状态|

若已有预测但比较失败，先看错误涉及股票日期键、行情哈希还是 FT 逐月收益复算。compare.py 会逐项核对这些条件；可用原输出单独重跑比较。公开文件包括源码、协议与特征名；行情、持仓、个股预测和模型权重由运行者自行提供。
