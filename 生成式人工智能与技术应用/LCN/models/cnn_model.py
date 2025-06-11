import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # 第一个卷积层：输入通道1，输出通道32，卷积核3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # 第二个卷积层：输入通道32，输出通道64，卷积核3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # 第三个卷积层：输入通道64，输出通道128，卷积核3x3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # 第一个卷积块
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 输出大小: 14x14
        
        # 第二个卷积块
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 输出大小: 7x7
        
        # 第三个卷积块
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 输出大小: 3x3
        
        # 展平
        x = x.view(-1, 128 * 3 * 3)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

    def train_step(self, batch, device):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        
        return loss
    
    def evaluate(self, dataloader, device):
        self.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy 