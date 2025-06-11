import torch
import torch.nn as nn
import numpy as np

class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.2):
        super(RNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch_size, seq_len, hidden_size]
        
        # 注意力权重计算
        attention_weights = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # 应用注意力权重
        context = torch.sum(attention_weights * lstm_out, dim=1)  # [batch_size, hidden_size]
        
        # 全连接层预测
        out = self.fc(context)  # [batch_size, 1]
        
        return out

    def train_step(self, batch, device):
        x, y = batch
        x, y = x.to(device), y.to(device)
        
        # 前向传播
        outputs = self(x)
        loss = nn.MSELoss()(outputs, y)
        
        return loss
    
    def evaluate(self, dataloader, device, threshold=0.1):
        self.eval()
        total_samples = 0
        correct_predictions = 0
        total_loss = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                outputs = self(x)
                
                # 计算相对误差
                relative_error = torch.abs(outputs - y) / torch.abs(y)
                correct_predictions += torch.sum(relative_error < threshold).item()
                total_samples += y.size(0)
                
                # 计算MSE损失
                loss = nn.MSELoss()(outputs, y)
                total_loss += loss.item()
        
        accuracy = 100 * correct_predictions / total_samples
        avg_loss = total_loss / len(dataloader)
        
        return accuracy, avg_loss 