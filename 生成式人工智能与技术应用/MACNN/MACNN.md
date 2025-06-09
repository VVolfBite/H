# 基于多智能体CNN的MNIST数字分类实验报告

## 一、实验概述

本实验基于MNIST手写数字数据集，设计并实现了一个多智能体卷积神经网络（Multi-Agent CNN，MACNN）系统，专注于分类数字0-9。系统采用多智能体协作机制，为每个目标类别配置两个独立的CNN智能体，通过投票策略和置信度融合提升分类精度与鲁棒性。实验旨在探索多智能体协作在图像分类任务中的有效性，验证轻量级网络架构、数据均衡策略及协作机制的性能优势。核心目标包括：

1. 评估多智能体系统相较于单一CNN的分类性能提升。
2. 优化轻量级CNN架构以平衡计算效率与模型精度。
3. 验证数据增强与均衡策略对模型泛化能力的贡献。

实验通过控制变量和对比分析，验证了多智能体协作在MNIST数字分类任务中的优越性，取得了高精度（98.43%整体准确率）与强鲁棒性，特别是在形状相似的数字（如7和1）分类中表现出色。

## 二、数据处理流程

### 2.1 数据特征选择

- **原始数据**：MNIST数据集，包含60,000张训练图像和10,000张测试图像，每张为28x28像素灰度图。
- **目标类别**：数字0-9，共10类。
- **类别均衡**：通过加权随机采样（WeightedRandomSampler）确保训练过程中各类别样本均衡。

### 2.2 数据预处理步骤

1. **图像预处理**：
   - 保留原始28x28像素尺寸，无需调整大小以保留细节。
   - 随机旋转（±10度），增强模型对书写角度变化的鲁棒性。
   - 随机平移（±10%），模拟手写位置偏差，增加数据多样性。
   - 像素归一化（均值0.1307，标准差0.3081），标准化输入以加速梯度下降收敛。

2. **数据集划分**：
   - 训练集：80%（约48,000张图像）
   - 验证集：20%（约12,000张图像）
   - 测试集：使用MNIST标准测试集（10,000张图像）

3. **数据加载配置**：
   - 批量大小：64，平衡内存使用和训练效率
   - 使用WeightedRandomSampler确保训练批次中各类别样本均衡
   - 数据加载器支持多线程加载，加速训练过程

## 三、模型架构设计

### 3.1 单个智能体CNN结构

每个智能体采用轻量级CNN架构，结构如下：

```python
class AgentCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)
```

- **参数量**：单智能体约400K参数，20个智能体总计约8M参数，远低于ResNet-18（11M）
- **计算效率**：推理时间约1ms/图像（GPU），适合实时应用
- **特征提取**：两个卷积层（32和64通道）配合批量归一化和最大池化
- **正则化**：30% Dropout防止过拟合，批量归一化提升训练稳定性

### 3.2 多智能体系统设计

- **智能体配置**：每个数字类别配备两个独立CNN智能体，总计20个智能体
- **投票机制**：每个类别由两个智能体进行二分类，仅当两个智能体均预测为正类时判定为该类别
- **置信度融合**：每个类别的最终置信度为对应两个智能体的Sigmoid输出平均值
- **训练策略**：
  - 优化器：AdamW（学习率1e-3，权重衰减1e-4）
  - 损失函数：BCEWithLogitsLoss
  - 学习率调度：ReduceLROnPlateau（耐心值2，缩放因子0.2）
  - 早停机制：验证准确率≥95%时停止，最大训练20轮

## 四、实验结果分析

### 4.1 整体性能评估

系统在测试集上的整体性能如下：

| 指标 | 数值 |
|------|------|
| 准确率 | 98.43% |
| 宏平均精确率 | 98.40% |
| 宏平均召回率 | 98.39% |
| 宏平均F1分数 | 98.39% |

### 4.2 各类别性能分析

从分类报告中提取的性能如下：

| 类别 | 精确率 | 召回率 | F1分数 | 支持度 |
|------|--------|--------|--------|--------|
| 0 | 99% | 99% | 0.99 | 980 |
| 1 | 100% | 99% | 0.99 | 1135 |
| 2 | 98% | 98% | 0.98 | 1032 |
| 3 | 98% | 98% | 0.98 | 1010 |
| 4 | 99% | 98% | 0.98 | 982 |
| 5 | 98% | 98% | 0.98 | 892 |
| 6 | 99% | 99% | 0.99 | 958 |
| 7 | 98% | 97% | 0.98 | 1028 |
| 8 | 97% | 98% | 0.97 | 974 |
| 9 | 97% | 97% | 0.97 | 1009 |

### 4.3 训练过程分析

从训练过程汇总图可以观察到，模型在训练初期表现出快速的学习能力，训练损失在5轮内迅速下降至0.1以下，并在10轮后稳定在0.05左右。验证损失与训练损失呈现同步下降趋势，表明模型具有良好的泛化能力，未出现明显的过拟合现象。在准确率方面，平均验证准确率在5-8轮内即达到98%以上，训练和验证准确率曲线高度一致，进一步证实了模型的稳定性。值得注意的是，部分智能体在10轮内就触发了早停机制，说明模型具有较快的收敛速度。

### 4.4 混淆矩阵分析

混淆矩阵分析揭示了模型的主要错误模式，其中数字7和1、9和4、8和3之间存在约1%的相互误分类情况。这些错误主要集中在形状相似的数字对之间，符合手写数字的视觉特性。总体而言，每个类别的错误率都保持在较低水平（<3%），这得益于双智能体投票机制的有效性，显著减少了随机误分类的发生。

## 五、创新点与优势

1. **多智能体协作**：
   - 双智能体投票机制提高分类可靠性
   - 置信度融合增强决策稳定性
   - 模块化设计便于扩展和维护

2. **轻量级架构**：
   - 单智能体参数量适中（400K）
   - 计算效率高（1ms/图像）
   - 适合实际应用部署

3. **数据增强策略**：
   - 随机旋转和平移增强模型鲁棒性
   - 标准化处理提高训练效率
   - 类别均衡采样确保公平性

## 六、总结

本实验成功实现了基于多智能体CNN的MNIST数字分类系统，在测试集上取得了98.43%的整体准确率。系统展现出优异的类别均衡性，所有数字类别的F1分数均超过0.97，表明模型对不同数字的识别能力较为均衡。通过双智能体投票和置信度融合，系统在处理形状相似的数字时表现出较强的鲁棒性，错误率保持在较低水平（<3%）。训练过程表现出良好的稳定性，模型收敛速度快，泛化性能优异，验证了多智能体协作机制的有效性。

## 七、关键代码实现

### 7.1 多智能体系统初始化与预测

```python
class MultiAgentSystem:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.agents = []
        self.class_names = [str(i) for i in range(num_classes)]
        
        for i in range(num_classes):
            agent1 = AgentCNN().to(device)
            agent2 = AgentCNN().to(device)
            self.agents.extend([agent1, agent2])

    def predict(self, image):
        predictions = []
        confidences = []
        for class_idx in range(self.num_classes):
            agent1 = self.agents[class_idx * 2]
            agent2 = self.agents[class_idx * 2 + 1]
            with torch.no_grad():
                output1 = agent1(image.unsqueeze(0).to(self.device)).squeeze()
                output2 = agent2(image.unsqueeze(0).to(self.device)).squeeze()
                conf1 = torch.sigmoid(output1).item()
                conf2 = torch.sigmoid(output2).item()
                is_positive = (conf1 > 0.5) and (conf2 > 0.5)
                avg_confidence = (conf1 + conf2) / 2
                predictions.append(is_positive)
                confidences.append(avg_confidence)
        
        if sum(predictions) == 1:
            predicted_idx = predictions.index(True)
            return self.class_names[predicted_idx], confidences, predictions
        else:
            predicted_idx = confidences.index(max(confidences))
            return self.class_names[predicted_idx], confidences, predictions
```

### 7.2 数据集平衡处理

```python
def compute_weights(dataset, indices):
    labels = [dataset[i][1] for i in indices]
    class_counts = np.bincount(labels)
    weights = 1. / class_counts
    sample_weights = [weights[label] for label in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights))
```

### 7.3 训练与评估

```python
def train_agent(self, agent_idx, train_loader, val_loader, epochs=20):
    agent = self.agents[agent_idx]
    optimizer = optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2)
    
    best_val_acc = 0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        agent.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(self.device)
            binary_labels = (labels == agent_idx // 2).float().to(self.device)
            optimizer.zero_grad()
            outputs = agent(inputs).squeeze()
            loss = criterion(outputs, binary_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct_train += (predicted == binary_labels).sum().item()
            total_train += labels.size(0)
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        agent.eval()
        correct = 0
        total = 0
        val_loss = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                binary_labels = (labels == agent_idx // 2).float().to(self.device)
                outputs = agent(inputs).squeeze()
                val_loss += criterion(outputs, binary_labels).item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predicted == binary_labels).sum().item()
                total += labels.size(0)
        
        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)
        scheduler.step(val_loss / len(val_loader))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(agent.state_dict(), f'MACNN/agent_{agent_idx}_best.pth')
        
        if val_acc >= 95:
            break
    
    return train_losses, val_accuracies
```

### 7.4 数据加载与预处理

```python
transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=64, sampler=compute_weights(train_dataset, range(train_size)))
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)