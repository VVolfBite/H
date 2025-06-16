from typing import Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from torchvision.datasets import MNIST
from config import ModelConfig
import logging

logger = logging.getLogger(__name__)

def load_mnist_data(config: ModelConfig) -> Tuple[DataLoader, DataLoader]:
    """加载MNIST数据"""
    try:
        # 数据转换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # 加载训练集和测试集
        train_dataset = MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = MNIST('./data', train=False, transform=transform)
        
        # 划分训练集和验证集
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
        
        return train_loader, val_loader
    except Exception as e:
        logger.error(f"加载MNIST数据失败: {str(e)}")
        raise

def load_stock_data(data_path: str, config: ModelConfig) -> Tuple[DataLoader, DataLoader, StandardScaler]:
    """加载股票数据"""
    try:
        # 读取数据
        df = pd.read_csv(data_path)
        
        # 数据预处理
        df = df.dropna()
        df = df.sort_values('date')
        
        # 特征工程
        features = ['open', 'high', 'low', 'close', 'volume', 'money', 'change']
        X = df[features].values
        y = df['close'].values
        
        # 数据标准化
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # 创建序列数据
        X_seq, y_seq = [], []
        for i in range(len(X) - config.sequence_length):
            X_seq.append(X[i:i + config.sequence_length])
            y_seq.append(y[i + config.sequence_length])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X_seq)
        y_tensor = torch.FloatTensor(y_seq).view(-1, 1)
        
        # 划分训练集和验证集
        train_size = int(0.8 * len(X_tensor))
        X_train, X_val = X_tensor[:train_size], X_tensor[train_size:]
        y_train, y_val = y_tensor[:train_size], y_tensor[train_size:]
        
        # 创建数据集
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
        
        return train_loader, val_loader, scaler
    except Exception as e:
        logger.error(f"加载股票数据失败: {str(e)}")
        raise 