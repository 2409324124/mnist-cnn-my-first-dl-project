"""
inference.py — 推理模块
职责: 加载模型、定义预处理、暴露 predict() 接口
不涉及任何 Web/UI 逻辑
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np

# ── 模型结构（与训练脚本保持一致）───────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ── 预处理（与训练时一致）───────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ── 单例模型（模块第一次被 import 时加载一次）────────────────────────────
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model  = SimpleCNN().to(_device)
_model.load_state_dict(
    torch.load("model.pth", map_location=_device, weights_only=True)
)
_model.eval()
print(f"[inference] 模型加载成功，设备: {_device}")


# ── 公开接口 ─────────────────────────────────────────────────────────────
def predict(pil_image: Image.Image) -> list[float]:
    """
    输入: PIL 灰度图（任意尺寸，白底黑字 或 黑底白字均可）
    输出: 长度为 10 的概率列表，索引对应数字 0-9
    """
    # 确保灰度
    gray = pil_image.convert("L")
    # 自动判断背景色，保证送入模型时是"黑底白字"（MNIST 风格）
    arr = np.array(gray)
    if arr.mean() > 128:          # 白底黑字 → 反转
        gray = ImageOps.invert(gray)

    tensor = _transform(gray).unsqueeze(0).to(_device)   # [1,1,28,28]
    with torch.no_grad():
        probs = F.softmax(_model(tensor), dim=1).squeeze().tolist()
    return probs                  # [p0, p1, ..., p9]
