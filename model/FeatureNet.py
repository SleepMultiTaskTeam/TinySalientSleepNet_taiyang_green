import torch.nn as nn
import torch
'''
Single channel EEG feature extraction
'''
def FeatureNet(s_freq, filters=128, dropout=0.5):
    model = nn.Sequential(
        nn.Conv1d(1, filters, kernel_size=s_freq//2, stride=s_freq//4),
        nn.BatchNorm1d(filters),
        nn.ReLU(),
        nn.MaxPool1d(8, 8),
        nn.Dropout(dropout),
        nn.Conv1d(filters, filters, kernel_size=8, stride=1, padding=4),
        nn.BatchNorm1d(filters),
        nn.ReLU(),
        nn.Conv1d(filters, filters, kernel_size=8, stride=1, padding=4),
        nn.BatchNorm1d(filters),
        nn.ReLU(),
        nn.Conv1d(filters, filters, kernel_size=8, stride=1, padding=4),
        nn.BatchNorm1d(filters),
        nn.ReLU(),
        nn.MaxPool1d(4, 4),
        nn.Dropout(dropout),
        nn.Flatten()
    )
    return model


'''
Multi-channel EEG feature extraction
'''
class FeatureNet_MC(nn.Module):
    def __init__(self, s_freq, filters=128):
        super(FeatureNet_MC, self).__init__()
        self.tiny_model = FeatureNet(s_freq, filters)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,5),
        )

    def forward(self, X):
        B, C, T = X.shape
        X = X.reshape(B * C, 1, T)
        H = self.tiny_model(X)
        out = self.classifier(H.reshape(B, C, -1))
        return out.unsqueeze(-1)
    
# # LSTM
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMClassifier, self).__init__()
        # 定义LSTM层
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # 定义全连接层，将LSTM的输出特征映射到五个类别
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 转换输入形状为 (batch, time_step, input_size)
        x = x.permute(0, 2, 1)  # 将 (batch, channel, time_step) 转换为 (batch, time_step, channel)
        
        # LSTM的输出是 (output, (h_n, c_n))
        _, (h_n, _) = self.lstm(x)  # 取LSTM的最后一个隐藏状态
        # h_n的形状为 (num_layers, batch, hidden_size)，取最后一层的隐藏状态
        out = h_n[-1, :, :]  # 形状为 (batch, hidden_size)
        out = self.fc(out)   # 映射到5个类别
        out = out.unsqueeze(-1) # (batch, 5, input_epoch_len)
        return out


if __name__ == '__main__':
    # net = FeatureNet_MC(100)
    # 参数设置
    input_size = 2         # 输入特征数 (channel 数)
    hidden_size = 64       # LSTM的隐藏层大小，可以根据需要调整
    num_layers = 2         # LSTM层数
    output_size = 5        # 输出类别数（五分类）

    # 创建模型
    model = LSTMClassifier(input_size, hidden_size, num_layers, output_size)

    # 生成一个随机输入张量，形状为 (batch, channel, time_step)
    batch_size = 64
    time_step = 3000
    x = torch.randn(batch_size, 2, time_step)  # 原始输入形状为 (64, 2, 3000)

    # 前向传播
    output = model(x)  # 输出形状为 (batch, output_size)
    print(output.shape)  # 输出应为 (64, 5)
    # X= torch.zeros((64,2,3000))
    # out = model(X)
    # print(out.shape)
