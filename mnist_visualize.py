# 首先导入必要的库
import matplotlib.pyplot as plt
# 添加这一行：支持中文显示（解决字体警告）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # Windows 自带中文字体
plt.rcParams['axes.unicode_minus'] = False

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

print("开始导入模块...")

# 数据加载（完整，包括训练集和测试集）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("加载训练集和测试集...")
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=False,  # 已手动放好文件，用 False
    transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    download=False, 
    transform=transform
)

# 关键：定义 train_loader（训练用）
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 测试用 loader
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print("数据集加载完成！")

# 模型定义
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model = SimpleCNN().to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练（5 个 epoch）
print("开始训练...")
train_losses = []
for epoch in range(5):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"第 {epoch+1} 轮完成，平均损失: {avg_loss:.4f}")

print("训练完成！")

# 测试 + 收集预测结果（用于混淆矩阵）
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
print(f"测试准确率: {accuracy:.2f}%")

# 绘制 Loss 曲线
plt.figure(figsize=(8, 5))
plt.plot(range(1, 6), train_losses, marker='o', color='b')
plt.title('训练 Loss 曲线')
plt.xlabel('轮次 (Epoch)')
plt.ylabel('平均损失 (Loss)')
plt.grid(True)
plt.show()

# 漂亮混淆矩阵
print("\n绘制混淆矩阵...")
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.title('混淆矩阵（行:真实标签，列:预测标签）')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.show()

print("所有可视化完成！")