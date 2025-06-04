# LSTM 股票预测模型实验报告

## 一、实验概述

本实验旨在基于历史股票市场数据，构建一个能够预测未来收盘价的时间序列模型。模型核心采用 LSTM（长短期记忆）网络，能够有效处理具有时间依赖性的金融数据。与传统的单特征预测不同，本模型结合了多个交易指标（如开盘价、最高价、最低价、成交量等），以期提升模型对趋势和波动的感知能力。

## 二、数据处理流程

### 2.1 数据特征选择
模型输入使用了如下多个交易日特征，覆盖价格、成交行为和涨跌幅变化等多个维度：

```python
feature_cols = ['open', 'close', 'low', 'high', 'volume', 'money', 'change']
```

目标列为收盘价（'close'）。

### 2.2 数据预处理步骤
为满足 LSTM 模型对时间序列输入格式的要求，我们进行以下数据准备工作：

1. 标准化处理：使用 StandardScaler 分别对输入特征和目标值进行标准化，保证不同量纲的特征具有相近的分布。

2. 时间窗口构建：以滑动窗口方式将每连续 30 天的特征数据作为一个输入序列，预测第 31 天的收盘价。

3. 数据集划分：将数据按时间顺序划分为训练集和测试集，比例为 80%:20%。

关键处理代码：
```python
# 标准化
features = feature_scaler.fit_transform(df[feature_cols].values)
labels = label_scaler.fit_transform(df[target_col].values.reshape(-1, 1))

# 序列构造
def create_sequences(features, labels, seq_len):
    x, y = [], []
    for i in range(len(features) - seq_len):
        x.append(features[i:i+seq_len])
        y.append(labels[i+seq_len])
    return np.array(x), np.array(y)
```

## 三、模型架构设计

本模型采用多层 LSTM 网络架构，包含输入层、LSTM 层、全连接层和 Dropout 层：

```python
class LstmRNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.4
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)
```

## 四、训练设置

训练过程中使用了以下参数配置：

| 参数项 | 数值 |
|--------|------|
| 批大小（batch_size） | 64 |
| 学习率（learning rate） | 0.001 |
| 优化器（optimizer） | Adam |
| 损失函数 | 相对误差损失 |
| 训练轮数（epochs） | 30 |

## 五、实验结果分析

### 5.1 训练过程
模型训练过程中，损失值从初始的 0.835087 快速下降，在第 20 轮达到了 0.215455 的较低水平，之后趋于稳定，最终在第 30 轮收敛到 0.236176。这表明模型训练过程稳定，学习效果良好。

### 5.2 误差指标
模型在测试集上的主要评估指标：

```
MSE（均方误差）：12886.36
RMSE（均方根误差）：113.52
MAE（平均绝对误差）：59.27
```

这些指标反映了模型在实际预测中的表现。考虑到股票价格的量级（约在 2000-3000 元区间），RMSE 为 113.52 表示预测值与实际值的平均偏差在价格的 4% 左右，MAE 为 59.27 表示绝对误差的平均值相对较小，说明模型具有一定的预测能力。

### 5.3 可视化结果

#### 训练损失曲线
训练过程中损失值整体呈下降趋势，在前期快速下降后逐渐趋于稳定，表明模型训练充分：
![训练损失曲线](training_loss.png)

#### 预测结果对比
预测结果与实际价格走势的对比图显示，模型能够较好地捕捉价格变动趋势：
![预测结果对比](prediction_results.png)

## 六、结论分析

### 6.1 模型表现
- 训练过程表现稳定，损失值持续下降并最终收敛
- 在市场平稳期，预测误差较小，MAE 保持在 60 元以内
- 模型对价格的整体趋势有较好的把握能力

### 6.2 误差原因
- 金融市场的固有不确定性和波动性
- 部分极端行情下的预测偏差较大
- 预测值与实际值的偏差在剧烈波动时期会有所增大

## 七、环境配置

- Python 3.x
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

代码在支持 CUDA 的 GPU 环境下运行，可显著提升训练速度。 