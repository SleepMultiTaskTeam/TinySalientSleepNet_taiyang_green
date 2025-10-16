import torch
from torch import nn
import torch.nn.functional as F
# from torchsummary import summary


class segmentClassifier(nn.Module):
    def __init__(self, kernel_size, filters, sleep_epoch_len):
        super(segmentClassifier, self).__init__()
        self.pool = nn.AvgPool1d(sleep_epoch_len)
        self.convDown = nn.Conv1d(filters[0], 5, kernel_size, padding=(kernel_size-1) // 2)
        self.conv1x1 = nn.Conv1d(filters[0], filters[0], kernel_size=1)
        # self.fc = nn.Linear(5+116, 64)
        # self.fc2 = nn.Linear(64, 5)

    def forward(self, x, x_yasa=None):
        x = self.conv1x1(x)
        x = torch.tanh(x)
        x = self.pool(x)
        x = self.convDown(x)# (batch,5,10)
        # if x_yasa is not None:
        #     x = torch.cat((x, x_yasa), 1)
        #     batch_size = x.size(0)
        #     x = x.permute(0,2,1).reshape(-1, 5+116)
        #     x = F.relu(self.fc(x))
        #     x = self.fc2(x)
        #     x = x.reshape(batch_size, -1, 5).permute(0, 2, 1)
        # x = F.softmax(x, dim=1)
        return x


if __name__ == '__main__':
    # 参数
    kernel_size = 5
    filters = [4, 8, 16, 32, 64]
    pooling_size = [10, 10, 10, 3]
    sleep_epoch_len = 3000
    sequence_len = 20
    stride = 3
    layer_num = 5
    bottleneck_filters = [i // 2 for i in filters]

    #创建模型与测试
    testmodel = segmentClassifier(kernel_size, bottleneck_filters, sleep_epoch_len)
    input = torch.ones(4, 2, 60000)
    output = testmodel(input)
    print(input)
    print(output)
    print(output.shape)
    # testmodel.cuda()
    # summary(testmodel, (2, 60000))
