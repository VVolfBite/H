import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

print("Starting data loading...")

# =======================
# 1. 读取与处理数据
# =======================
try:
    df = pd.read_csv("stock_dataset_2.csv")
    print(f"Successfully loaded data with shape: {df.shape}")
    print("\nFirst few rows of data:")
    print(df.head())
    print("\nData columns:", df.columns.tolist())
    df['date'] = pd.to_datetime(df['date'])
except Exception as e:
    print(f"Error reading data file: {e}")
    exit(1)

feature_cols = ['open', 'close', 'low', 'high', 'volume', 'money', 'change']
target_col = 'label'

# 验证数据列是否存在
missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
if missing_cols:
    print(f"Missing columns in dataset: {missing_cols}")
    exit(1)

print("\nFeature statistics before scaling:")
print(df[feature_cols].describe())

print("\nStarting feature scaling...")
# 特征和标签归一化
feature_scaler = StandardScaler()
label_scaler = StandardScaler()
features = feature_scaler.fit_transform(df[feature_cols].values)
labels = label_scaler.fit_transform(df[target_col].values.reshape(-1, 1))
print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

print("\nFeature statistics after scaling:")
print(pd.DataFrame(features, columns=feature_cols).describe())

# =======================
# 2. 构建时间序列
# =======================
print("\nCreating sequences...")
SEQ_LEN = 30

def create_sequences(features, labels, seq_len):
    x, y = [], []
    for i in range(len(features) - seq_len):
        x.append(features[i:i+seq_len])
        y.append(labels[i+seq_len])
    return np.array(x), np.array(y)

X, y = create_sequences(features, labels, SEQ_LEN)
print(f"Sequences created. X shape: {X.shape}, y shape: {y.shape}")

# 划分训练与测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
print(f"\nTrain set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# 转为 Tensor
X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

# =======================
# 3. 构建模型
# =======================
print("\nBuilding model...")

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
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.fc1(lstm_out[:, -1, :])
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 相对误差损失函数
class RelativeErrorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        return torch.mean(torch.abs((pred - target) / (target + 1e-8)))

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# 创建模型实例
model = LstmRNN(
    input_size=len(feature_cols),
    hidden_size=64,
    num_layers=1
).to(device)

criterion = RelativeErrorLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

# 打印模型参数
print("\nModel parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# =======================
# 4. 模型训练
# =======================
print("\nStarting training...")
loss_list = []
EPOCHS = 30
best_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    batch_count = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1
        
        # 打印第一个batch的预测结果
        if epoch == 0 and batch_count == 1:
            print("\nFirst batch prediction sample:")
            print("Input shape:", xb[0].shape)
            print("True value:", yb[0].item())
            print("Predicted value:", pred[0].item())
    
    avg_loss = total_loss / len(train_loader)
    loss_list.append(avg_loss)
    
    # 更新学习率
    scheduler.step(avg_loss)
    
    # 保存最佳模型
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), 'best_model.pth')
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}")

# =======================
# 5. 可视化训练损失
# =======================
print("\nPlotting training loss...")
plt.figure(figsize=(10, 6))
plt.plot(loss_list)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Relative Error Loss")
plt.grid(True)
plt.savefig('training_loss.png')
plt.close()

# =======================
# 6. 测试集评估
# =======================
print("\nEvaluating on test set...")
model.eval()
predictions = []
true_values = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        pred = model(xb)
        predictions.append(pred.cpu().numpy())
        true_values.append(yb.numpy())

predictions = np.concatenate(predictions, axis=0)
true_values = np.concatenate(true_values, axis=0)

# 反向转换预测结果
predictions = label_scaler.inverse_transform(predictions)
true_values = label_scaler.inverse_transform(true_values)

# 打印一些预测样本
print("\nPrediction samples (first 5):")
for i in range(5):
    print(f"Sample {i+1}:")
    print(f"True value: {true_values[i][0]:.2f}")
    print(f"Predicted value: {predictions[i][0]:.2f}")
    print(f"Absolute error: {abs(true_values[i][0] - predictions[i][0]):.2f}")

# 计算评估指标
mse = np.mean((predictions - true_values) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(predictions - true_values))

print(f"\nTest Set Metrics:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# 绘制预测结果
print("\nPlotting prediction results...")
plt.figure(figsize=(12, 6))
plt.plot(true_values, label="True", alpha=0.7)
plt.plot(predictions, label="Predicted", alpha=0.7)
plt.legend()
plt.title("True vs Predicted (Test Set)")
plt.grid(True)
plt.savefig('prediction_results.png')
plt.close()

# 保存预测结果到CSV
print("\nSaving prediction results to CSV...")
results_df = pd.DataFrame({
    'True_Value': true_values.flatten(),
    'Predicted_Value': predictions.flatten(),
    'Absolute_Error': np.abs(true_values.flatten() - predictions.flatten())
})

# 添加日期列
test_dates = df['date'].values[-len(true_values):]
results_df['Date'] = test_dates

# 重新排列列顺序
results_df = results_df[['Date', 'True_Value', 'Predicted_Value', 'Absolute_Error']]

# 保存到CSV
results_df.to_csv('prediction_results.csv', index=False)
print("Prediction results saved to 'prediction_results.csv'")

print("\nDone!")
