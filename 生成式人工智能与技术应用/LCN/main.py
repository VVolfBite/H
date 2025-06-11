import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models.cnn_model import CNNModel
from models.rnn_model import RNNModel
from langchain_integration import LangChainIntegration
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from dotenv import load_dotenv

def train_model(model, train_loader, val_loader, optimizer, device, epochs=50, model_type="CNN"):
    """训练模型并返回训练历史"""
    history = {
        'train_loss': [],
        'val_accuracy': []
    }
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model.train_step(batch, device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 验证
        if model_type == "CNN":
            val_accuracy = model.evaluate(val_loader, device)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%')
            history['val_accuracy'].append(val_accuracy)
        else:  # RNN
            val_accuracy, val_loss = model.evaluate(val_loader, device)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%')
            history['val_accuracy'].append(val_accuracy)
        
        history['train_loss'].append(total_loss/len(train_loader))
    
    return history

def prepare_mnist_data():
    """准备MNIST数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

def prepare_time_series_data(data_path, sequence_length=10):
    """准备时间序列数据"""
    # 读取数据
    df = pd.read_csv(data_path)
    
    # 数据预处理
    scaler = MinMaxScaler()
    values = scaler.fit_transform(df['close'].values.reshape(-1, 1))
    
    # 创建序列
    X, y = [], []
    for i in range(len(values) - sequence_length):
        X.append(values[i:i+sequence_length])
        y.append(values[i+sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 创建数据加载器
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader, scaler

def plot_training_history(history, model_type):
    """绘制训练历史"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'])
    plt.title(f'{model_type} Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'])
    plt.title(f'{model_type} Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig(f'{model_type.lower()}_training_history.png')
    plt.close()

def main():
    # 加载环境变量
    load_dotenv()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 准备数据
    mnist_train_loader, mnist_test_loader = prepare_mnist_data()
    time_series_train_loader, time_series_test_loader, scaler = prepare_time_series_data(
        'data/stock_dataset_2.csv'
    )
    
    # 初始化模型
    cnn_model = CNNModel().to(device)
    rnn_model = RNNModel().to(device)
    
    # 定义优化器
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)
    
    # 训练模型
    print("Training CNN model...")
    cnn_history = train_model(
        cnn_model, mnist_train_loader, mnist_test_loader,
        cnn_optimizer, device, epochs=20, model_type="CNN"
    )
    
    print("\nTraining RNN model...")
    rnn_history = train_model(
        rnn_model, time_series_train_loader, time_series_test_loader,
        rnn_optimizer, device, epochs=50, model_type="RNN"
    )
    
    # 绘制训练历史
    plot_training_history(cnn_history, "CNN")
    plot_training_history(rnn_history, "RNN")
    
    # 评估模型
    cnn_accuracy = cnn_model.evaluate(mnist_test_loader, device)
    rnn_accuracy, rnn_loss = rnn_model.evaluate(time_series_test_loader, device)
    
    print(f"\nFinal Results:")
    print(f"CNN Accuracy: {cnn_accuracy:.2f}%")
    print(f"RNN Accuracy: {rnn_accuracy:.2f}%")
    
    # 使用LangChain生成分析报告
    langchain_integration = LangChainIntegration()
    
    # 示例预测
    sample_image, _ = next(iter(mnist_test_loader))
    sample_sequence, _ = next(iter(time_series_test_loader))
    
    with torch.no_grad():
        cnn_pred = cnn_model(sample_image[0:1].to(device)).argmax().item()
        rnn_pred = scaler.inverse_transform(
            rnn_model(sample_sequence[0:1].to(device)).cpu().numpy()
        )[0][0]
    
    # 生成分析报告
    analysis = langchain_integration.analyze_results(
        cnn_result=cnn_pred,
        cnn_accuracy=cnn_accuracy,
        rnn_result=rnn_pred,
        rnn_accuracy=rnn_accuracy
    )
    
    # 生成完整实验报告
    training_history = {
        'cnn': cnn_history,
        'rnn': rnn_history
    }
    
    test_results = {
        'cnn_accuracy': cnn_accuracy,
        'rnn_accuracy': rnn_accuracy,
        'rnn_loss': rnn_loss
    }
    
    report = langchain_integration.generate_report(
        training_history=str(training_history),
        test_results=str(test_results)
    )
    
    # 保存报告
    with open('experiment_report.txt', 'w', encoding='utf-8') as f:
        f.write(analysis)
        f.write('\n\n完整实验报告：\n')
        f.write(report)

if __name__ == "__main__":
    main() 