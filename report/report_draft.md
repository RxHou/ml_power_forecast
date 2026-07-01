# 家庭电力消耗多变量时间序列预测实验报告

作者：侯嘉睿

研究领域：机器学习与智能电力负荷预测

Github 链接：待提交代码后填写

## 1. 问题介绍

随着智能家居和物联网技术的发展，家庭电力消耗预测可以帮助居民理解用电行为、发现异常用电，并为电网负荷调度提供参考。本实验基于 UCI Machine Learning Repository 的 Individual Household Electric Power Consumption 数据集，对法国某家庭 2006 年 12 月至 2010 年 11 月的分钟级用电记录进行建模。

原始数据包含全屋有功功率、无功功率、电压、电流强度以及多个分表能耗。由于课程任务要求预测未来每天的总有功功率，本实验首先将分钟级数据聚合为天级数据：`global_active_power`、`global_reactive_power`、`sub_metering_1`、`sub_metering_2`、`sub_metering_3` 按天求和，`voltage`、`global_intensity` 按天求平均，并根据公式计算剩余分表能耗：

`sub_metering_remainder = global_active_power * 1000 / 60 - (sub_metering_1 + sub_metering_2 + sub_metering_3)`

预测任务设置为多变量多步预测：使用过去 90 天的特征序列预测未来 90 天和未来 365 天的 `global_active_power`。两个预测长度分别训练独立模型，避免模型参数混用。数据划分采用时间顺序划分，前 80% 滑动窗口用于训练和验证，后 20% 用于测试，避免随机划分导致未来信息泄漏。

## 2. 模型

### 2.1 LSTM 模型

LSTM 适合处理时间序列中的顺序依赖。本实验将过去 90 天的多变量特征输入两层 LSTM，取最后时刻隐藏状态作为历史序列表示，再通过全连接层一次性输出未来 `H` 天预测值，其中 `H` 为 90 或 365。

伪代码：

```text
X = past 90-day multivariate sequence
h = LSTM(X).last_hidden_state
y_hat = MLP(h)
```

### 2.2 Transformer 模型

Transformer 使用自注意力机制建模序列中不同时间位置之间的关系。实验中先将输入特征映射到隐藏维度，加入正弦位置编码，再经过 Transformer Encoder，最后对时间维度进行平均池化并输出未来功率曲线。

伪代码：

```text
Z = Linear(X) + PositionalEncoding
E = TransformerEncoder(Z)
h = MeanPool(E)
y_hat = MLP(h)
```

### 2.3 改进模型：CNN-Transformer

自提出模型采用 CNN-Transformer 结构。局部卷积层先提取相邻日期之间的短期变化、周周期波动和局部趋势，再由 Transformer Encoder 捕捉更长范围的依赖关系。最后使用注意力池化聚合不同日期的信息。该结构的动机是：家庭用电同时具有局部连续性和季节性/长期依赖，单纯 LSTM 或 Transformer 可能分别偏向顺序记忆或全局关系，而卷积前端能为 Transformer 提供更稳定的局部模式表示。

伪代码：

```text
Z_local = Conv1D(X)
Z = Z_local + PositionalEncoding
E = TransformerEncoder(Z)
h = AttentionPool(E)
y_hat = MLP(h)
```

## 3. 结果与分析

本实验使用 MSE 和 MAE 作为评价指标。每个模型在每种预测长度下使用 5 个随机种子独立训练，报告平均值和标准差。

### 3.1 短期预测结果：未来 90 天

| 模型 | MSE mean | MSE std | MAE mean | MAE std |
| --- | ---: | ---: | ---: | ---: |
| LSTM | 157627.27 | 1762.05 | 293.33 | 2.65 |
| Transformer | 171443.32 | 6126.89 | 316.48 | 9.24 |
| CNN-Transformer | 174413.49 | 8781.44 | 316.40 | 11.43 |

截图位置：

- `outputs/figures/lstm_h90_seed42.png`
- `outputs/figures/transformer_h90_seed42.png`
- `outputs/figures/cnn-transformer_h90_seed42.png`

### 3.2 长期预测结果：未来 365 天

| 模型 | MSE mean | MSE std | MAE mean | MAE std |
| --- | ---: | ---: | ---: | ---: |
| Transformer | 162738.04 | 1374.06 | 302.78 | 0.84 |
| LSTM | 164592.31 | 2681.69 | 303.85 | 1.52 |
| CNN-Transformer | 166347.88 | 4707.92 | 306.25 | 5.01 |

截图位置：

- `outputs/figures/lstm_h365_seed42.png`
- `outputs/figures/transformer_h365_seed42.png`
- `outputs/figures/cnn-transformer_h365_seed42.png`

### 3.3 对比分析

短期预测中，模型主要需要捕捉最近用电水平、周周期和局部趋势，因此 LSTM 与 CNN-Transformer 预计表现较稳定。长期预测中，误差通常会增大，因为未来 365 天包含季节变化和生活习惯变化，模型需要更强的长期依赖建模能力。若 CNN-Transformer 的长期预测优于基础 Transformer，可以说明卷积层提取的局部模式有助于后续注意力建模；若性能没有提升，则需要分析卷积前端是否过度平滑、训练样本是否不足、长期输出维度是否导致优化难度上升。

## 4. 讨论

本实验的主要难点包括：原始数据为分钟级且存在缺失值，需要先进行聚合和插值；时间序列划分不能随机打乱，否则会引入未来信息；长期预测输出维度高，训练更不稳定。实验采用多随机种子训练并报告标准差，以评估模型稳定性。

改进模型的创新点在于结合局部卷积特征提取和 Transformer 长期依赖建模。该方法不改变预测任务设定，仍使用相同输入与输出窗口，因此可以与 LSTM 和基础 Transformer 进行公平比较。后续可进一步加入天气变量、节假日信息、分解趋势与季节项，或采用直接多步预测与自回归预测相结合的策略。

## 参考文献

[1] UCI Machine Learning Repository. Individual household electric power consumption Dataset.

[2] Hochreiter, S., & Schmidhuber, J. Long Short-Term Memory. Neural Computation, 1997.

[3] Vaswani, A., et al. Attention Is All You Need. NeurIPS, 2017.

[4] 使用 ChatGPT/Codex 辅助撰写报告文字和组织代码结构，实验代码与结果由本项目运行生成。
