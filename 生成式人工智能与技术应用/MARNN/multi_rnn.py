import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import matplotlib as mpl

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

print("Loading data...")

# =======================
# 1. Data Loading and Preprocessing
# =======================
try:
    df = pd.read_csv("RNN/stock_dataset_2.csv")
    print(f"Data loaded successfully, shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
    df['date'] = pd.to_datetime(df['date'])
except Exception as e:
    print(f"Error loading data file: {e}")
    exit(1)

feature_cols = ['open', 'close', 'low', 'high', 'volume', 'money', 'change']
target_col = 'label'

# Verify data columns
missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
if missing_cols:
    print(f"Missing columns in dataset: {missing_cols}")
    exit(1)

print("\nFeature statistics before scaling:")
print(df[feature_cols].describe())

print("\nStarting feature scaling...")
# Feature and label normalization
feature_scaler = StandardScaler()
label_scaler = StandardScaler()
features = feature_scaler.fit_transform(df[feature_cols].values)
labels = label_scaler.fit_transform(df[target_col].values.reshape(-1, 1))
print(f"Feature shape: {features.shape}, Label shape: {labels.shape}")

# =======================
# 2. Sequence Creation
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

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
print(f"\nTraining set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Convert to Tensor
X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

# =======================
# 3. Model Definitions
# =======================
print("\nBuilding models...")

class SingleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.4
        )
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.fc1(lstm_out[:, -1, :])
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MultiLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm1 = SingleLSTM(input_size, hidden_size=64)
        self.lstm2 = SingleLSTM(input_size, hidden_size=128)
        self.lstm3 = SingleLSTM(input_size, hidden_size=256)
        
    def forward(self, x):
        pred1 = self.lstm1(x)
        pred2 = self.lstm2(x)
        pred3 = self.lstm3(x)
        return (pred1 + pred2 + pred3) / 3

# =======================
# 4. Load Trained Models and Generate Predictions
# =======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Load multi-agent model
multi_model = MultiLSTM(input_size=len(feature_cols)).to(device)
multi_model.load_state_dict(torch.load('best_multi_rnn_model.pth'))
multi_model.eval()

# Create and train single-agent model
single_model = SingleLSTM(input_size=len(feature_cols)).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(single_model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
)

# Train single model
print("\nTraining single LSTM model...")
EPOCHS = 100
best_loss = float('inf')

for epoch in range(EPOCHS):
    single_model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = single_model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(single_model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(single_model.state_dict(), 'best_single_lstm_model.pth')
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}")

# Load best single model
single_model.load_state_dict(torch.load('best_single_lstm_model.pth'))
single_model.eval()

# =======================
# 5. Generate Predictions
# =======================
print("\nGenerating predictions...")

def get_predictions(model, test_loader):
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
    
    return predictions, true_values

# Get predictions
single_preds, true_values = get_predictions(single_model, test_loader)
multi_preds, _ = get_predictions(multi_model, test_loader)

# Inverse transform predictions
single_preds = label_scaler.inverse_transform(single_preds)
multi_preds = label_scaler.inverse_transform(multi_preds)
true_values = label_scaler.inverse_transform(true_values)

# Create comparison DataFrame
results_df = pd.DataFrame({
    'Date': df['date'].values[-len(true_values):],
    'True_Value': true_values.flatten(),
    'Single_LSTM_Prediction': single_preds.flatten(),
    'Multi_LSTM_Prediction': multi_preds.flatten(),
    'Single_LSTM_Error': np.abs(true_values.flatten() - single_preds.flatten()),
    'Multi_LSTM_Error': np.abs(true_values.flatten() - multi_preds.flatten())
})

# Save results
results_df.to_csv('model_comparison_results.csv', index=False)

# =======================
# 6. Visualization and Analysis
# =======================
# Plot predictions
plt.figure(figsize=(15, 8))
plt.plot(results_df['Date'], results_df['True_Value'], label='真实值', alpha=0.7)
plt.plot(results_df['Date'], results_df['Single_LSTM_Prediction'], label='单LSTM预测', alpha=0.7)
plt.plot(results_df['Date'], results_df['Multi_LSTM_Prediction'], label='多LSTM预测', alpha=0.7)
plt.title('预测结果对比')
plt.xlabel('日期')
plt.ylabel('数值')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('prediction_comparison.png')
plt.close()

# Calculate metrics
single_mse = np.mean((single_preds - true_values) ** 2)
single_rmse = np.sqrt(single_mse)
single_mae = np.mean(np.abs(single_preds - true_values))

multi_mse = np.mean((multi_preds - true_values) ** 2)
multi_rmse = np.sqrt(multi_mse)
multi_mae = np.mean(np.abs(multi_preds - true_values))

print("\nPerformance Metrics:")
print("\nSingle LSTM:")
print(f"MSE: {single_mse:.4f}")
print(f"RMSE: {single_rmse:.4f}")
print(f"MAE: {single_mae:.4f}")

print("\nMulti LSTM:")
print(f"MSE: {multi_mse:.4f}")
print(f"RMSE: {multi_rmse:.4f}")
print(f"MAE: {multi_mae:.4f}")

# Plot error distribution
plt.figure(figsize=(12, 6))
sns.kdeplot(data=results_df['Single_LSTM_Error'], label='单LSTM误差')
sns.kdeplot(data=results_df['Multi_LSTM_Error'], label='多LSTM误差')
plt.title('误差分布对比')
plt.xlabel('绝对误差')
plt.ylabel('密度')
plt.legend()
plt.grid(True)
plt.savefig('error_distribution.png')
plt.close()

print("\n分析完成！结果已保存至：")
print("1. model_comparison_results.csv")
print("2. prediction_comparison.png")
print("3. error_distribution.png") 