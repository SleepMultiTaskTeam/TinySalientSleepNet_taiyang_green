
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
T = 1079
L = 120 * T  # 129600


# bottleneck
class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, strides):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               padding=padding,
                               stride=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.pool1 = nn.MaxPool1d(kernel_size=kernel_size, padding=padding, stride=strides)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.pool1(out)
        out = self.relu(out)
        return out

# (inputs.shape[0], max_seq_len, embed_dim)

def PositionalEncoding(max_seq_len, embed_dim, len_inputs):
    positional_encoding = np.array([[
        [np.sin(pos / np.power(10000, 2 * i / embed_dim)) if i % 2 == 0 else
         np.cos(pos / np.power(10000, 2 * i / embed_dim))
         for i in range(embed_dim)]
        for pos in range(max_seq_len)] for i in range(len_inputs)])

    return torch.tensor(positional_encoding).float()


class LFiveStaging(nn.Module):
    def __init__(self, dmodel=512, head=4, inputlen=180,dropout=.0):
        super(LFiveStaging, self).__init__()
        # 4通道
        self.bn1 = nn.BatchNorm1d(2)
        # yasa 特征
        self.bn2 = nn.BatchNorm1d(116)
        # 4通道
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=1, stride=1)
        self.model1 = BottleNeck(64, 64, 11, 5, 4)
        self.model2 = BottleNeck(64, 256, 7, 3, 6)
        self.model3 = BottleNeck(256, 512, 5, 2, 5)
        self.model4 = BottleNeck(512, 512, 5, 2, 5)
        self.model5 = BottleNeck(512, 512, 5, 2, 5)
        self.FC1 = nn.Linear(512, 512)  
        encoder_layer = nn.TransformerEncoderLayer(d_model=dmodel, nhead=head, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.FC2 = nn.Linear(512+116, 256)
        self.FC3 = nn.Linear(256, 5)
        self.dropout = nn.Dropout(dropout)
        # 降采样之后
        self.max_seq_len = inputlen
        self.pos_embedding = PositionalEncoding(max_seq_len=self.max_seq_len, embed_dim=dmodel, len_inputs=1).cuda()
        
        
    def forward(self, inputs, inputs1):
        # inputs = inputs.permute(0,2,1)
        inputs = self.bn1(inputs)
        # yasa 特征
        inputs1 = self.bn2(inputs1)
        output = self.conv1(inputs)
        output = self.model1(output)
        output = self.model2(output)
        output = self.model3(output)
        output = self.model4(output)
        output = self.model5(output)
        output = output.transpose(1, 2)
        # output = output.reshape(output.shape[0], -1)
        output = self.FC1(output)
        # (batch, epoch_len, embed_dim)
        output = self.dropout(output)
        output = output + self.pos_embedding
        output = self.transformer_encoder(output)
        # yasa特征
        inputs1 = inputs1.permute(0,2,1)
        combined = torch.cat((output, inputs1), dim=-1)
        # output = self.FC2(output)
        output = self.FC2(combined)
        output = self.FC3(output)
        output = output.permute(0,2,1)
        # print(output.shape)
        return output

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # test_data = torch.randn(8, 149, 180*30*100).cuda()
    # test_y = torch.randint(0, 5, (8, 180)).cuda()
    model = LFiveStaging().cuda()
    # model.zero_grad()
    # loss_func = torch.nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # model.train()
    # outcome = model(test_data)
    # loss = loss_func(outcome,test_y.long())
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    print('LF:'+str(sum([param.nelement() for param in model.parameters()])))
    # LF:22781809
    # print(outcome.shape)
# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
