# -*- coding: utf-8 -*-
"""
训练股票收盘价 LSTM（精简版）
"""
import os, pandas as pd, numpy as np, torch, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

# ─────────────────────────── 超参数 / 路径 ──────────────────────────
CSV_PATH        = "./data/stock_dataset_2.csv"
SAVE_PATH       = "./models/saved/stock_rnn.pth"
SEQ_LEN         = 30
BATCH_SIZE      = 64
EPOCHS          = 30
LR              = 1e-3
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURES        = ['open', 'close', 'low', 'high', 'volume', 'money', 'change']
TARGET          = 'label'

# ───────────────────────────── 数据 ────────────────────────────────
df = pd.read_csv(CSV_PATH)
fs, ls = df[FEATURES].values, df[TARGET].values.reshape(-1, 1)

f_scaler, l_scaler = StandardScaler(), StandardScaler()
fs = f_scaler.fit_transform(fs)
ls = l_scaler.fit_transform(ls)

def make_seq(x, y, n):
    xs, ys = [], []
    for i in range(len(x)-n):
        xs.append(x[i:i+n]); ys.append(y[i+n])
    return np.array(xs), np.array(ys)

X, y = make_seq(fs, ls, SEQ_LEN)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)

tr_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                     torch.tensor(y_tr, dtype=torch.float32)),
                       batch_size=BATCH_SIZE, shuffle=True)
te_loader = DataLoader(TensorDataset(torch.tensor(X_te, dtype=torch.float32),
                                     torch.tensor(y_te, dtype=torch.float32)),
                       batch_size=BATCH_SIZE)

# ───────────────────────────── 网络 ────────────────────────────────
class StockLSTM(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, 64, 1, batch_first=True)
        self.fc   = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

model = StockLSTM(len(FEATURES)).to(DEVICE)

# ───────────────────────────── 训练 ────────────────────────────────
def train_model(save_path: str = SAVE_PATH, verbose: bool = True):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    crit = lambda p,t: torch.mean(torch.abs((p - t)/(t+1e-8)))
    opt  = torch.optim.Adam(model.parameters(), lr=LR)
    best = float('inf'); loss_hist = []

    for ep in range(1, EPOCHS+1):
        model.train(); l_sum = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
            l_sum += loss.item()
        avg = l_sum/len(tr_loader); loss_hist.append(avg)
        if verbose:
            print(f"[RNN] Epoch {ep}/{EPOCHS}  Loss: {avg:.4f}")

        if avg < best:
            best = avg
            torch.save(model.state_dict(), save_path)
            if verbose:
                print(f"[RNN] 🏆 New best loss {best:.4f}, model saved → {save_path}")

    # 画 loss 曲线（可选）
    plt.figure(figsize=(6,3))
    plt.plot(loss_hist); plt.title("Train Loss"); plt.tight_layout()
    plt.savefig("stock_rnn_loss.png"); plt.close()

def test_model(verbose: bool = True):
    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            xb = xb.to(DEVICE)
            preds.append(model(xb).cpu().numpy()); trues.append(yb.numpy())
    preds = np.concatenate(preds); trues = np.concatenate(trues)
    preds = l_scaler.inverse_transform(preds)
    trues = l_scaler.inverse_transform(trues)

    mse  = np.mean((preds-trues)**2)
    mae  = np.mean(np.abs(preds-trues))
    if verbose:
        print(f"[RNN] Test MSE:{mse:.2f}  MAE:{mae:.2f}")

# ───────────────────────────── CLI ─────────────────────────────────
if __name__ == "__main__":
    train_model()
    test_model()
