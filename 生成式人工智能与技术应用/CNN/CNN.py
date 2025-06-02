# -*- coding: utf-8 -*-
"""
猫狗二分类CNN训练脚本（PyTorch版）
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import pandas as pd

# 自定义ImageFolder，跳过损坏图片
def pil_loader_with_skip(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except Exception as e:
        print(f"跳过损坏图片: {path}")
        raise e

class MyImageFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = pil_loader_with_skip(path)
        except Exception:
            # 随机返回一张合法图片，防止dataloader崩溃
            return self.__getitem__((index + 1) % len(self.samples))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

# 1. 数据加载与预处理
print("Starting data loading...")
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

data_dir = './data/PetImages'
dataset = MyImageFolder(root=data_dir, transform=transform)
print(f"Successfully loaded dataset with {len(dataset)} images.")

# 划分训练集和验证集（8:2）
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(f"Train set size: {train_size}, Validation set size: {val_size}")

batch_size = 32
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
classes = dataset.classes  # ['Cat', 'Dog']
print(f"Classes: {classes}")

# 2. 定义CNN模型
print("\nBuilding model...")
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 2)  # 二分类
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# 3. 训练与验证
def train_model(model, trainloader, valloader, criterion, optimizer, device, epochs=10):
    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_loss = running_loss / total
        train_acc = correct / total
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # 验证
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    return train_loss_list, train_acc_list, val_loss_list, val_acc_list

# 4. 主流程
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10
    print(f"\nStarting training for {epochs} epochs...")
    train_loss, train_acc, val_loss, val_acc = train_model(model, trainloader, valloader, criterion, optimizer, device, epochs)

    # 5. 保存模型
    torch.save(model.state_dict(), 'cat_dog_cnn.pth')
    print("\nModel saved as 'cat_dog_cnn.pth'")

    # 6. 绘制训练曲线
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(train_acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    print("\nTraining curves saved as 'training_history.png'")

    print(f"\n最终训练准确率: {train_acc[-1]:.4f}")
    print(f"最终验证准确率: {val_acc[-1]:.4f}")

    # 7. 预测打印
    print("\nPredicting on validation set...")
    model.eval()
    predictions = []
    true_values = []
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.append(predicted.cpu().numpy())
            true_values.append(labels.numpy())
    predictions = np.concatenate(predictions, axis=0)
    true_values = np.concatenate(true_values, axis=0)
    print("\nPrediction samples (first 5):")
    for i in range(5):
        print(f"Sample {i+1}:")
        print(f"True class: {classes[true_values[i]]}")
        print(f"Predicted class: {classes[predictions[i]]}")

    # 保存预测结果到CSV
    results_df = pd.DataFrame({
        'True_Class': [classes[i] for i in true_values],
        'Predicted_Class': [classes[i] for i in predictions]
    })
    results_df.to_csv('prediction_results.csv', index=False)
    print("\nPrediction results saved to 'prediction_results.csv'")

    # 8. 结果图片
    plt.figure(figsize=(12, 6))
    plt.plot(true_values, label="True", alpha=0.7)
    plt.plot(predictions, label="Predicted", alpha=0.7)
    plt.legend()
    plt.title("True vs Predicted (Validation Set)")
    plt.grid(True)
    plt.savefig('prediction_results.png')
    plt.close()
    print("\nPrediction results saved as 'prediction_results.png'")

    print("\nDone!")
