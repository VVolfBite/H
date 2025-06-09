import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import csv
from datetime import datetime
import pandas as pd

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 创建保存目录
os.makedirs('MACNN', exist_ok=True)

# 定义单个Agent的CNN网络结构
class AgentCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 修改为单通道输入
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # 调整全连接层大小以适应MNIST图像尺寸
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class MultiAgentSystem:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.agents = []
        # 使用数字类别名称
        self.class_names = [str(i) for i in range(num_classes)]
        
        # 为每个类别创建两个Agent
        for i in range(num_classes):
            print(f"为数字 {self.class_names[i]} 创建两个Agent:")
            print(f"创建 Agent {i*2} (第一个Agent)")
            agent1 = AgentCNN().to(device)
            print(f"创建 Agent {i*2+1} (第二个Agent)")
            agent2 = AgentCNN().to(device)
            self.agents.extend([agent1, agent2])
    
    def eval(self):
        """将所有Agent设置为评估模式"""
        for agent in self.agents:
            agent.eval()
    
    def train(self):
        """将所有Agent设置为训练模式"""
        for agent in self.agents:
            agent.train()
    
    def prepare_binary_data(self, dataset, target_class):
        # 准备二分类数据
        binary_labels = []
        for i in range(len(dataset)):
            label = dataset[i][1]
            # 将目标类别标记为1，其他类别标记为0
            binary_labels.append(1 if label == target_class else 0)
        return binary_labels
    
    def train_agent(self, agent_idx, train_loader, val_loader, epochs=20):  # 减少训练轮次
        print(f"\n{'='*50}")
        print(f"开始训练 Agent {agent_idx} - 负责识别类别: {self.class_names[agent_idx // 2]}")
        print(f"{'='*50}")
        
        agent = self.agents[agent_idx]
        # 使用 AdamW 优化器提升训练速度
        optimizer = optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        # 调整学习率调度器的参数
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2)
        
        best_val_acc = 0
        train_losses = []
        val_accuracies = []
        val_losses = []  # 添加验证损失记录
        
        # 创建训练日志文件
        log_file = f'MACNN/agent_{agent_idx}_training_log.csv'
        
        with open(log_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Val Accuracy', 'Learning Rate'])
        
        for epoch in range(epochs):
            print(f"\n训练轮次 {epoch+1}/{epochs}")
            print("-" * 30)
            
            # 训练阶段
            agent.train()
            total_loss = 0
            correct_train = 0
            total_train = 0
            batch_count = len(train_loader)
            
            for batch_idx, (inputs, labels) in enumerate(train_loader, 1):
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
                
                # 减少进度打印频率
                if batch_idx % 20 == 0 or batch_idx == batch_count:
                    current_train_acc = 100 * correct_train / total_train
                    print(f"批次进度: [{batch_idx}/{batch_count}] "
                          f"当前损失: {loss.item():.4f} "
                          f"当前训练准确率: {current_train_acc:.2f}%")
            
            avg_loss = total_loss / len(train_loader)
            train_acc = 100 * correct_train / total_train
            train_losses.append(avg_loss)
            
            # 验证阶段
            print("\n开始验证...")
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
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)  # 记录验证损失
            
            # 更新学习率
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录训练日志
            with open(log_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, avg_loss, avg_val_loss, val_acc, current_lr])
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(agent.state_dict(), f'MACNN/agent_{agent_idx}_best.pth')
                print(f"\n发现新的最佳模型！验证准确率提升至: {val_acc:.2f}%")
            
            print(f'\n第 {epoch+1} 轮训练总结:')
            print(f'训练损失: {avg_loss:.4f}, 训练准确率: {train_acc:.2f}%')
            print(f'验证损失: {avg_val_loss:.4f}, 验证准确率: {val_acc:.2f}%')
            print(f'当前学习率: {current_lr:.6f}')
            print(f'历史最佳验证准确率: {best_val_acc:.2f}%')
            
            # 如果验证准确率达到目标，提前结束训练
            if val_acc >= 95:
                print(f"\n提前结束训练！已达到目标准确率: {val_acc:.2f}%")
                break
        
        return train_losses, val_accuracies, val_losses  # 返回验证损失
    
    def predict(self, image):
        # 获取每对Agent的预测结果
        predictions = []
        confidences = []
        
        for class_idx in range(self.num_classes):
            # 获取当前类别的两个Agent
            agent1 = self.agents[class_idx * 2]
            agent2 = self.agents[class_idx * 2 + 1]
            
            # 获取两个Agent的预测结果
            with torch.no_grad():
                output1 = agent1(image.unsqueeze(0).to(self.device)).squeeze()
                output2 = agent2(image.unsqueeze(0).to(self.device)).squeeze()
                
                conf1 = torch.sigmoid(output1).item()
                conf2 = torch.sigmoid(output2).item()
                
                # 只有当两个Agent都预测为正类时，才认为属于该类
                is_positive = (conf1 > 0.5) and (conf2 > 0.5)
                # 使用两个Agent的平均置信度
                avg_confidence = (conf1 + conf2) / 2
                
                predictions.append(is_positive)
                confidences.append(avg_confidence)
        
        # 如果有且仅有一个类别被两个Agent都预测为正类
        if sum(predictions) == 1:
            predicted_idx = predictions.index(True)
            return self.class_names[predicted_idx], confidences, predictions
        else:
            # 如果没有类别或有多个类别被预测为正类，选择平均置信度最高的
            predicted_idx = confidences.index(max(confidences))
            return self.class_names[predicted_idx], confidences, predictions

def plot_training_curves(train_losses, val_accuracies, val_losses, agent_idx, class_name, agent_type, save_dir='MACNN'):
    """绘制训练曲线"""
    plt.figure(figsize=(15, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='训练损失')
    plt.plot(val_losses, 'r--', label='验证损失')  # 添加验证损失曲线
    plt.title(f'Agent {agent_idx} (数字{class_name}-{agent_type}) 损失曲线')
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.grid(True)
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, 'g-', label='验证准确率')
    plt.title(f'Agent {agent_idx} (数字{class_name}-{agent_type}) 验证准确率')
    plt.xlabel('训练轮次')
    plt.ylabel('准确率 (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/agent_{agent_idx}_training.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_final_results(y_true, y_pred, class_names, save_dir='MACNN'):
    """绘制最终评估结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 混淆矩阵
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.title("混淆矩阵")
    plt.savefig(f'{save_dir}/confusion_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 每个类别的准确率条形图
    plt.figure(figsize=(15, 6))
    class_accuracies = []
    for i in range(len(class_names)):
        mask = np.array(y_true) == i
        if np.sum(mask) > 0:
            accuracy = np.mean(np.array(y_pred)[mask] == i)
            class_accuracies.append(accuracy)
        else:
            class_accuracies.append(0)
    
    plt.bar(class_names, class_accuracies)
    plt.title('各类别准确率')
    plt.xlabel('数字类别')
    plt.ylabel('准确率')
    plt.ylim(0, 1)
    for i, v in enumerate(class_accuracies):
        plt.text(i, v + 0.01, f'{v:.2%}', ha='center')
    plt.savefig(f'{save_dir}/class_accuracies_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 训练过程汇总图
    plt.figure(figsize=(15, 10))
    
    # 读取所有Agent的训练日志
    all_train_losses = []
    all_val_accuracies = []
    
    for i in range(len(class_names) * 2):
        log_file = f'{save_dir}/agent_{i}_training_log.csv'
        if os.path.exists(log_file):
            df = pd.read_csv(log_file)
            all_train_losses.append(df['Train Loss'].values)
            all_val_accuracies.append(df['Val Accuracy'].values)
    
    # 绘制平均训练损失
    plt.subplot(2, 1, 1)
    mean_train_loss = np.mean(all_train_losses, axis=0)
    std_train_loss = np.std(all_train_losses, axis=0)
    epochs = range(1, len(mean_train_loss) + 1)
    plt.plot(epochs, mean_train_loss, 'b-', label='平均训练损失')
    plt.fill_between(epochs, 
                     mean_train_loss - std_train_loss,
                     mean_train_loss + std_train_loss,
                     alpha=0.2)
    plt.title('所有智能体的平均训练损失')
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.grid(True)
    plt.legend()
    
    # 绘制平均验证准确率
    plt.subplot(2, 1, 2)
    mean_val_acc = np.mean(all_val_accuracies, axis=0)
    std_val_acc = np.std(all_val_accuracies, axis=0)
    plt.plot(epochs, mean_val_acc, 'r-', label='平均验证准确率')
    plt.fill_between(epochs,
                     mean_val_acc - std_val_acc,
                     mean_val_acc + std_val_acc,
                     alpha=0.2)
    plt.title('所有智能体的平均验证准确率')
    plt.xlabel('训练轮次')
    plt.ylabel('准确率 (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_summary_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 数据预处理和增强
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载MNIST数据集
    print("正在加载MNIST数据集...")
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                             download=True, transform=transform_train)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True, transform=transform_test)
    
    print("数据集加载完成！")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 使用所有10个数字类别
    selected_classes = list(range(10))
    num_classes = len(selected_classes)
    
    # 筛选数据
    train_indices = [i for i, (_, label) in enumerate(train_dataset) if label in selected_classes]
    test_indices = [i for i, (_, label) in enumerate(test_dataset) if label in selected_classes]
    
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    # 将训练集按照8:2的比例划分为训练集和验证集
    train_size = int(0.8 * len(train_subset))
    val_size = len(train_subset) - train_size
    train_subset, val_subset = random_split(train_subset, [train_size, val_size])
    
    print(f"划分后的训练集大小: {len(train_subset)}")
    print(f"验证集大小: {len(val_subset)}")
    
    # 计算训练集的类别权重
    train_labels = [train_dataset[i][1] for i in train_indices[:train_size]]
    class_counts = np.bincount(train_labels)
    weights = 1. / class_counts
    sample_weights = [weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # 创建数据加载器
    train_loader = DataLoader(train_subset, batch_size=64, sampler=sampler)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
    
    # 创建多智能体系统
    system = MultiAgentSystem(num_classes, device)
    
    # 训练每个Agent
    for i in range(num_classes * 2):  # 现在是20个Agent
        class_idx = i // 2
        agent_type = "第一个" if i % 2 == 0 else "第二个"
        print(f"\n训练数字 {system.class_names[class_idx]} 的{agent_type} Agent (Agent {i})...")
        train_losses, val_accuracies, val_losses = system.train_agent(i, train_loader, val_loader)
        
        # 使用新的绘图函数
        plot_training_curves(train_losses, val_accuracies, val_losses, i, 
                           system.class_names[class_idx], agent_type)
    
    # 在测试集上评估系统性能
    print("\n开始最终测试集评估...")
    system.eval()
    y_true = []
    y_pred = []
    
    # 创建预测结果CSV文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions_file = f'MACNN/predictions_{timestamp}.csv'
    
    with open(predictions_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['真实类别', '预测类别']
        # 添加每个类别的两个Agent的置信度
        for class_name in system.class_names:
            header.extend([
                f'{class_name}_Agent1_置信度',
                f'{class_name}_Agent2_置信度',
                f'{class_name}_平均置信度',
                f'{class_name}_预测结果'
            ])
        writer.writerow(header)
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                for i in range(len(images)):
                    predicted_class, all_confidences, predictions = system.predict(images[i])
                    true_class = system.class_names[labels[i]]
                    
                    # 准备行数据
                    row = [true_class, predicted_class]
                    
                    # 添加每个类别的详细信息
                    for class_idx in range(system.num_classes):
                        agent1_conf = torch.sigmoid(system.agents[class_idx * 2](images[i].unsqueeze(0))).item()
                        agent2_conf = torch.sigmoid(system.agents[class_idx * 2 + 1](images[i].unsqueeze(0))).item()
                        avg_conf = (agent1_conf + agent2_conf) / 2
                        is_positive = (agent1_conf > 0.5) and (agent2_conf > 0.5)
                        
                        row.extend([
                            f"{agent1_conf:.4f}",
                            f"{agent2_conf:.4f}",
                            f"{avg_conf:.4f}",
                            "是" if is_positive else "否"
                        ])
                    
                    writer.writerow(row)
                    
                    y_pred.append(system.class_names.index(predicted_class))
                    y_true.append(labels[i].item())
    
    # 使用新的结果可视化函数
    plot_final_results(y_true, y_pred, system.class_names)
    
    # 保存分类报告
    report = classification_report(y_true, y_pred, target_names=system.class_names)
    with open(f'MACNN/classification_report_{timestamp}.txt', 'w') as f:
        f.write(report)
    
    # 打印最终结果
    print("\n" + "="*50)
    print("测试结果:")
    print("="*50)
    print("\n分类报告:")
    print(report)
    
    # 计算总体准确率
    accuracy = 100 * np.mean(np.array(y_true) == np.array(y_pred))
    print(f"\n系统整体准确率: {accuracy:.2f}%")
    print(f"\n详细预测结果已保存至: {predictions_file}")
    print(f"混淆矩阵已保存至: MACNN/confusion_matrix_{timestamp}.png")
    print(f"分类报告已保存至: MACNN/classification_report_{timestamp}.txt")
    print(f"训练过程汇总图已保存至: MACNN/training_summary_{timestamp}.png")
    print(f"各类别准确率图已保存至: MACNN/class_accuracies_{timestamp}.png")

if __name__ == "__main__":
    main() 