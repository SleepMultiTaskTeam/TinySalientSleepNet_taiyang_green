import torch.nn as nn
import torch

# 定义 CNN 模型
class SleepStageModel(nn.Module):
    def __init__(self):
        super(SleepStageModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2, stride=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2, stride=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(59904, 128)  # 根据输入的维度修改线性层的输入维度
        # self.bn4 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 5)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = nn.functional.leaky_relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = nn.functional.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = nn.functional.leaky_relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = torch.flatten(x, start_dim=1)
        x = nn.functional.relu(self.fc1(x))
        # x = nn.functional.relu(self.bn4(self.fc1(x)))
        # x = self.dropout(x)
        x = self.fc2(x)
        # x = x.view(-1, 1, 5)
        return x