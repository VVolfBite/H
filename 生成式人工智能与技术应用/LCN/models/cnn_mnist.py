# -*- coding: utf-8 -*-
"""
训练 MNIST 0‑9 十分类 CNN（精简版）
"""
import os, torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ────────────────────────────── 全局常量 ─────────────────────────────
BATCH_SIZE   = 64
EPOCHS       = 5
LR           = 1e-3
SAVE_PATH    = "./models/saved/mnist_cnn.pth"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────── 数据集 ──────────────────────────────
_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
_train = datasets.MNIST("./data", train=True, download=True, transform=_tf)
_test  = datasets.MNIST("./data", train=False, download=True, transform=_tf)
train_loader = DataLoader(_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(_test , batch_size=BATCH_SIZE)

# ────────────────────────────── 网络 ────────────────────────────────
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1, self.conv2, self.conv3 = [nn.Conv2d(i, o, 3, padding=1)
            for i, o in [(1, 32), (32, 64), (64, 128)]]
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1  = nn.Linear(128 * 3 * 3, 128)
        self.fc2  = nn.Linear(128, 10)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        for conv in (self.conv1, self.conv2, self.conv3):
            x = self.pool(F.relu(conv(x)))
        x = torch.flatten(x, 1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

model     = Net().to(DEVICE)
criterion  = nn.CrossEntropyLoss()
optimizer  = optim.Adam(model.parameters(), lr=LR)

# ────────────────────────────── 训练 / 测试 ─────────────────────────
def train_model(save_path: str = SAVE_PATH, verbose: bool = True):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_acc = 0.0
    for ep in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        if verbose:
            print(f"[CNN] Epoch {ep}/{EPOCHS}  Loss: {loss_sum/len(train_loader):.4f}")

        # 每个 epoch 做一次快速验证
        acc = test_model(verbose=False)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            if verbose:
                print(f"[CNN] 🏆 New best acc {acc:.2f}%, model saved → {save_path}")

def test_model(verbose: bool = True) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = torch.argmax(model(xb), 1)
            total   += yb.size(0)
            correct += (pred == yb).sum().item()
    acc = 100.0 * correct / total
    if verbose:
        print(f"[CNN] Test Accuracy: {acc:.2f}%")
    return acc

# ────────────────────────────── CLI ─────────────────────────────────
if __name__ == "__main__":
    train_model()
    test_model()
