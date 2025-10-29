import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# ==============================
# 设备自动检测：优先 CUDA，其次 Apple MPS，最后 CPU
# ==============================
if torch.cuda.is_available():
    device = torch.device("cuda")  # 使用 NVIDIA GPU（如 RTX 5070）
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")  # 使用 Apple 芯片的 Metal 后端
else:
    device = torch.device("cpu")  # 回退到 CPU
print(f"Using device: {device}")

# 对卷积/大规模算子开启 cuDNN 的算法搜索，可加速（仅 CUDA 有效）
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True


# ==============================
# 定义一个简单的前馈神经网络：784 -> 128 -> 10
# 适用于 MNIST（28x28 灰度图，类别 0~9）
# ==============================
class SimpleNN(nn.Module):
    # 负责定义有哪些层（layer），定义“网络的结构”
    def __init__(self):
        super(SimpleNN, self).__init__()
        # 全连接层1：输入 784（28*28 展平），负责特征提取（784 → 128），输出 128 个特征
        self.fc1 = nn.Linear(784, 128)
        # 全连接层2：输出层-负责分类输出（128 → 10） 输入 128，输出 10（10 个类别的 logits）
        self.fc2 = nn.Linear(128, 10)
        # 激活函数： ReLU 激活：缓解梯度消失，计算简单
        self.relu = nn.ReLU()

    # 定义数据流动的路径
    def forward(self, x):
        # 输入数据先经过 fc1 做线性变换
        x = self.fc1(x)  
        # 再经过 ReLU 激活函数
        x = self.relu(x)
        # 最后经过 fc2 输出分类结果
        x = self.fc2(x)  # 输出 logits（未做 softmax，交叉熵内部会处理）
        return x


# 实例化模型并放到目标设备（CPU/GPU）
model = SimpleNN().to(device)

# ==============================
# 损失函数与优化器
# ==============================
# CrossEntropyLoss = LogSoftmax + NLLLoss，输入为 logits，标签为类别索引
criterion = nn.CrossEntropyLoss()
# 随机梯度下降；学习率 0.01；可按需加入 momentum/weight_decay
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ==============================
# 数据集与数据加载
# ==============================
# ToTensor：将 PIL/Image 转为张量，并把像素从 [0,255] 归一化到 [0,1]
transform = transforms.ToTensor()

# 下载/加载 MNIST 训练集
train_dataset = datasets.MNIST(
    root="data", train=True, download=True, transform=transform
)

# CUDA 才开多进程和 pin_memory；MPS/CPU 统一设 0/False
use_workers = 2 if device.type == "cuda" else 0
pin_mem = True if device.type == "cuda" else False

# DataLoader：
# - batch_size: 每个小批次大小
# - shuffle: 每个 epoch 打乱数据
# - num_workers: 数据加载的子进程数（CPU 下设 0 避免多进程开销）
# - pin_memory: 仅对 CUDA 有意义，固定内存页，host->device 拷贝更快
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=use_workers,
    pin_memory=pin_mem,
)

# ==============================
# 训练循环
# ==============================
num_epochs = 5
model.train()  # 置为训练模式（影响如 Dropout/BatchNorm 的行为）
for epoch in range(num_epochs):
    running_loss = 0.0

    # 从 DataLoader 迭代取出 (inputs, labels)
    # - inputs: [B, 1, 28, 28] 的张量（B 为 batch 大小）
    # - labels: [B] 的长整型类别索引
    for inputs, labels in train_loader:
        # 把数据搬到与模型相同的设备
        # non_blocking=True 在 CUDA + pinned memory 时可减少同步等待
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 将图片展平为向量：从 [B, 1, 28, 28] -> [B, 784]
        inputs = inputs.view(inputs.size(0), -1)

        # 清空上一轮的梯度（否则梯度会累积）
        optimizer.zero_grad()

        # 前向传播：得到 logits，形状 [B, 10]
        outputs = model(inputs)

        # 计算交叉熵损失；labels 形状 [B]，值域为 0..9
        loss = criterion(outputs, labels)

        # 反向传播：基于当前 loss 计算各参数梯度
        loss.backward()

        # 根据梯度与学习率更新参数
        optimizer.step()

        # 累加损失用于监控
        running_loss += loss.item()

    # 每个 epoch 输出平均训练损失（总和 / 批次数）
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
