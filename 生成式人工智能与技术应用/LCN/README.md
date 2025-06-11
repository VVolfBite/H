# LangChain框架集成CNN和RNN模型

本项目使用LangChain框架连接图像分类（CNN）和回归（RNN）模型，实现高精度的混合模型系统。

## 项目特点

- CNN模型用于MNIST手写数字分类
- RNN模型用于时间序列数据预测
- LangChain框架集成大语言模型
- 模型准确率要求：
  - 整体命中率 ≥ 98%
  - 各模块准确率 ≥ 90%

## 系统要求

- Python 3.8+
- PyTorch 2.0+
- CUDA支持（可选，用于GPU加速）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 环境配置

1. 创建`.env`文件，添加以下内容：
```
HUGGINGFACE_API_TOKEN=your_token_here
```

2. 准备数据：
- MNIST数据集会自动下载
- 时间序列数据需放在`data/time_series_data.csv`

## 项目结构

```
.
├── models/
│   ├── cnn_model.py     # CNN模型定义
│   └── rnn_model.py     # RNN模型定义
├── data/                # 数据目录
├── main.py             # 主程序
├── langchain_integration.py  # LangChain集成
├── requirements.txt    # 项目依赖
└── README.md          # 项目说明
```

## 使用方法

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行程序：
```bash
python main.py
```

## 输出说明

程序运行后会生成：
- 训练历史图表（`cnn_training_history.png`和`rnn_training_history.png`）
- 实验报告（`experiment_report.txt`）
- 模型性能指标

## 模型架构

### CNN模型
- 3个卷积层
- 批归一化
- 最大池化
- Dropout正则化
- 2个全连接层

### RNN模型
- 2层LSTM
- 注意力机制
- Dropout正则化
- 全连接输出层

## LangChain集成

- 使用HuggingFace的LLaMA模型
- 提供模型分析和报告生成
- 支持交互式结果解释

## 性能优化

为达到要求的准确率：
1. CNN优化：
   - 使用批归一化
   - 添加Dropout
   - 优化网络结构

2. RNN优化：
   - 使用LSTM替代普通RNN
   - 添加注意力机制
   - 调整序列长度

## 注意事项

1. 确保有足够的计算资源（推荐使用GPU）
2. 正确设置HuggingFace API Token
3. 时间序列数据格式要求：CSV文件，包含'value'列

## 常见问题

1. GPU内存不足：
   - 减小批次大小
   - 减少模型层数

2. 训练不收敛：
   - 调整学习率
   - 增加训练轮数
   - 检查数据预处理

## 维护者

[您的名字/组织]

## 许可证

MIT License 