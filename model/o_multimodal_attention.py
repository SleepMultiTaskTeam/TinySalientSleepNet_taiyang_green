from torch import nn
import torch
from model.o_encoder import multi_layer_transformer
import torch.nn.functional as F
# from torchsummary import summary
from model.mysummary import summary

class channel_attention(nn.Module):
    def __init__(self, filter, with_res):
        super(channel_attention, self).__init__()
        self.avgPool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(filter, filter // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(filter // 2, filter, bias=False),
            nn.Sigmoid()
        )
        self.with_res = with_res
        if with_res:
            self.res_conv = nn.Sequential(
                nn.Conv1d(filter, filter, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(filter, filter, kernel_size=5, padding=2),
                nn.ReLU()
            )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avgPool(x).view(b, c)  # 去掉最后一维，相当于通道展平的步骤
        y = self.fc(y).view(b, c, 1)  # 还原维度
        y = y.expand_as(x)
        x = x * y

        if self.with_res:
            x2 = self.res_conv(x)
            x = x2 + x
        return x

class multimodal_megre_v9_type0(nn.Module):

    def __init__(self, filter):
        super(multimodal_megre_v9_type0, self).__init__()

        self.att = channel_attention(filter * 2, with_res=True)

        self.conv_smooth = nn.Sequential(
            nn.Conv1d(filter*2, filter, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(filter)
        )

    def forward(self, s1, s2):
        x = torch.cat((s1, s2), dim=1)
        x = self.att(x)
        x = self.conv_smooth(x)

        return x

class multimodal_megre_v9_type1(nn.Module):
    def __init__(self):
        super(multimodal_megre_v9_type1, self).__init__()

    def forward(self, s1, s2):
        # merge first
        x = s1 + s2
        return x

class multimodal_megre_v9_type2(nn.Module):
    def __init__(self):
        super(multimodal_megre_v9_type2, self).__init__()

    def forward(self, s1, s2):
        # merge first
        x = s1 * s2
        return x

class multimodal_megre_v9_type3(nn.Module):

    def __init__(self, filter):
        super(multimodal_megre_v9_type3, self).__init__()

        self.att1 = channel_attention(filter, with_res=True)
        self.att2 = channel_attention(filter, with_res=True)

        self.conv_smooth = nn.Sequential(
            nn.Conv1d(filter*2, filter, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(filter, filter, kernel_size=5, padding=2),
            nn.ReLU()
        )

    def forward(self, s1, s2):
        # channel att like single modal
        s1 = self.att1(s1)
        s2 = self.att2(s2)

        x = torch.cat((s1, s2), dim=1)
        x = self.conv_smooth(x)

        return x

class multimodal_megre_v9_type4(nn.Module):

    def __init__(self, filter):
        super(multimodal_megre_v9_type4, self).__init__()

        self.att = channel_attention(filter * 2, with_res=False)

    def forward(self, s1, s2):
        x = torch.cat((s1, s2), dim=1)
        x = self.att(x)
        b, c, l = x.shape
        x1 = x[:, :c//2, :]
        x2 = x[:, c//2:, :]
        x = x1 + x2

        return x

class multimodal_megre_v9_type5(nn.Module):

    def __init__(self, channel):
        super(multimodal_megre_v9_type5, self).__init__()
        self.att = channel_attention(channel * 2, with_res=True)

    def forward(self, s1, s2):
        x = torch.cat((s1, s2), dim=1)
        x = self.att(x)
        b, c, l = x.shape
        x1 = x[:, :c//2, :]
        x2 = x[:, c//2:, :]
        x = x1 * x2

        return x

class multimodal_megre_v9_type6(nn.Module):

    def __init__(self, channel):
        super(multimodal_megre_v9_type6, self).__init__()
        self.att = channel_attention(channel * 2, with_res=True)

    def forward(self, s1, s2):
        x = torch.cat((s1, s2), dim=1)
        x = self.att(x)
        b, c, l = x.shape
        x1 = x[:, :c//2, :]
        x2 = x[:, c//2:, :]
        a = x1 * x2
        x = x1 + x2 + a

        return x

class multimodal_megre_v9_type7(nn.Module):

    def __init__(self, channel):
        super(multimodal_megre_v9_type7, self).__init__()
        self.att = channel_attention(channel, with_res=True)

    def forward(self, s1, s2):
        x = s1 + s2
        x = self.att(x)

        return x

class multimodal_megre_v9_type8(nn.Module):

    def __init__(self, channel):
        super(multimodal_megre_v9_type8, self).__init__()
        self.att = channel_attention(channel, with_res=True)

    def forward(self, s1, s2):
        x = s1 * s2
        x = self.att(x)

        return x

class multimodal_megre_v9_type9(nn.Module):

    def __init__(self, channel):
        super(multimodal_megre_v9_type9, self).__init__()
        self.att = channel_attention(channel, with_res=True)

    def forward(self, s1, s2):
        a = s1 * s2
        x = s1 + s2 + a
        x = self.att(x)

        return x

class multimodal_megre_type11(nn.Module):

    def __init__(self, filter):
        super(multimodal_megre_type11, self).__init__()

        self.att1 = channel_attention(filter, with_res=True)
        self.att2 = channel_attention(filter, with_res=True)

        self.conv_smooth = nn.Sequential(
            nn.Conv1d(filter, filter, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(filter, filter, kernel_size=5, padding=2),
            nn.ReLU()
        )

    def forward(self, s1, s2):
        # channel att like single modal
        s1 = self.att1(s1)
        s2 = self.att2(s2)

        # x = torch.cat((s1, s2), dim=1)
        x = s1 * s2
        x = torch.add(x, s1)
        x = torch.add(x, s2)
        x = self.conv_smooth(x)

        return x

class multimodal_attention_layer_type0(nn.Module):
    def __init__(self, filter, downsize, two_stream):
        super(multimodal_attention_layer_type0, self).__init__()
        self.two_stream = two_stream

        self.conv1 = nn.Sequential(
            nn.Conv1d(filter*2, filter, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(filter, filter, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(filter, filter, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(filter, filter, kernel_size=5, padding=2),
            nn.ReLU()
        )

    def forward(self, s1, s2):
        # merge first
        if self.two_stream:
            x = torch.cat((s1, s2), dim=1)
            x = self.conv1(x)

        else:
            x = s1
            x = self.conv2(x)

        return x

class multimodal_attention_layer_type1(nn.Module):
    def __init__(self):
        super(multimodal_attention_layer_type1, self).__init__()

    def forward(self, s1, s2):
        # merge first
        x = s1 + s2
        return x



class only_modal_attention(nn.Module):
    def __init__(self, filter, downsize, two_stream):
        super(only_modal_attention, self).__init__()
        self.two_stream = two_stream
        self.avgPool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(filter, filter // downsize, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(filter // downsize, filter, bias=False),
            nn.Sigmoid()
        )
        #先cat的做att的话通道是两倍的
        self.fc_type2 = nn.Sequential(
            nn.Linear(2*filter, 2*filter // downsize, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(2*filter // downsize, 2*filter, bias=False),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv1d(filter*2, filter, kernel_size=5, padding=2),
            nn.Conv1d(filter, filter, kernel_size=5, padding=2)
        )

    def forward(self, s1, s2):
        # merge first
        if self.two_stream:
            x = torch.cat((s1, s2), dim=1)
            # channel attention
            b, c, _ = x.size()
            y = self.avgPool(x).view(b, c)  # 去掉最后一维，相当于通道展平的步骤
            y = self.fc_type2(y).view(b, c, 1)  # 还原维度
            y = y.expand_as(x)
            x = x * y
        else:
            x = s1
            # channel attention
            b, c, _ = x.size()
            y = self.avgPool(x).view(b, c)  # 去掉最后一维，相当于通道展平的步骤
            y = self.fc(y).view(b, c, 1)  # 还原维度
            y = y.expand_as(x)
            x = x * y

        x = self.conv(x)

        return x

class only_merge(nn.Module):
    def __init__(self, filter, downsize, two_stream):
        super(only_merge, self).__init__()
        self.two_stream = two_stream
        self.conv = nn.Sequential(
            nn.Conv1d(filter, filter, kernel_size=5, padding=2),
            nn.Conv1d(filter, filter, kernel_size=5, padding=2)
        )

    def forward(self, s1, s2):
        # merge first
        if self.two_stream:
            x = s1 * s2
            x = torch.add(x, s1)
            x = torch.add(x, s2)
        else:
            x = s1

        x = self.conv(x)

        return x


class multimodal_attention_layer(nn.Module):
    def __init__(self, filter, downsize, two_stream):
        super(multimodal_attention_layer, self).__init__()
        self.two_stream = two_stream
        self.avgPool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(filter, filter // downsize, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(filter // downsize, filter, bias=False),
            nn.Sigmoid()
        )

        self.conv = nn.Sequential(
            nn.Conv1d(filter, filter, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(filter, filter, kernel_size=5, padding=2),
            nn.ReLU()
        )

    def forward(self, s1, s2):
        # merge first
        if self.two_stream:
            x = s1 * s2
            # x = torch.add(x, s1)
            # x = torch.add(x, s2)
        else:
            x = s1

        # channel attention
        b, c, _ = x.size()
        y = self.avgPool(x).view(b, c) # 去掉最后一维，相当于通道展平的步骤
        y = self.fc(y).view(b, c, 1) # 还原维度
        y = y.expand_as(x)
        x = x * y
        # 整个残差
        x2 = self.conv(x)
        x = x2 + x
        return x

class multimodal_attention_layer_type2(nn.Module):
    def __init__(self, filter, downsize, two_stream):
        super(multimodal_attention_layer_type2, self).__init__()
        self.two_stream = two_stream
        self.avgPool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(filter, filter // downsize, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(filter // downsize, filter, bias=False),
            nn.Sigmoid()
        )

        #先cat的做att的话通道是两倍的
        self.fc_type2 = nn.Sequential(
            nn.Linear(2*filter, 2*filter // downsize, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(2*filter // downsize, 2*filter, bias=False),
            nn.Sigmoid()
        )

    def forward(self, s1, s2):
        if self.two_stream:
            x = torch.cat((s1, s2), dim=1)
            # channel attention first
            b, c, _ = x.size()
            y = self.avgPool(x).view(b, c) # 去掉最后一维，相当于通道展平的步骤
            y = self.fc_type2(y).view(b, c, 1) # 还原维度
            y = y.expand_as(x)
            x = x * y

            m1 = x[:, :c//2, :]
            m2 = x[:, c//2:, :]

            # merge
            x = m1 * m2
            x = torch.add(x, m1)
            x = torch.add(x, m2)

            return x

        else:
            # channel attention first
            x = s1
            b, c, _ = x.size()
            y = self.avgPool(x).view(b, c) # 去掉最后一维，相当于通道展平的步骤
            y = self.fc(y).view(b, c, 1) # 还原维度
            y = y.expand_as(x)
            x = x * y

            return x

# 先融合再trans
class multimodal_attention_layer_type3(nn.Module):
    def __init__(self, filter, downsize, two_stream):
        super(multimodal_attention_layer_type3, self).__init__()
        self.two_stream = two_stream
        self.avgPool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(filter, filter // downsize, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(filter // downsize, filter, bias=False),
            nn.Sigmoid()
        )

        self.conv = nn.Sequential(
            nn.Conv1d(filter, filter, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(filter, filter, kernel_size=5, padding=2),
            nn.ReLU()
        )

        self.trans = multi_layer_transformer(dim=filter, input_resolution=30000, num_heads=4, layerNum=6,
                                              window_size=300,
                                              shift_size=150)

    def forward(self, s1, s2):
        # merge first
        if self.two_stream:
            x = s1 * s2
            # x = torch.add(x, s1)
            # x = torch.add(x, s2)
        else:
            x = s1

        x = self.trans(x.transpose(-1, -2)).transpose(-1, -2)

        # # 整个残差
        # x2 = self.conv(x)
        # x = x2 + x
        return x



# 独立att，然后cat+卷积
# 此刻开始只为多模态设计

class multimodal_attention_layer_type4(nn.Module):
    def __init__(self, filter):
        super(multimodal_attention_layer_type4, self).__init__()
        self.att1 = channel_attention(filter)
        self.att2 = channel_attention(filter)

        self.conv = nn.Sequential(
            nn.Conv1d(filter*2, filter, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(filter, filter, kernel_size=5, padding=2),
            nn.ReLU()
        )

    def forward(self, s1, s2):
        s1 = self.att1(s1)
        s2 = self.att2(s2)

        x = torch.cat((s1, s2), dim=1)
        x = self.conv(x)

        return x


if __name__ == '__main__':
    # 参数
    kernel_size = 5
    filters = [16, 32, 64, 128, 256]
    pooling_size = [10, 8, 6, 5]  # 60000, 6000, 750, 125, 25
    sleep_epoch_len = 3000
    sequence_len = 20
    stride = 3
    layer_num = 5
    bottleneck_filters = [i // 2 for i in filters]
    downsize = 2 # 8/2剩下的channel数走fc

    # 构建模型与测试
    testmodel1 = multimodal_megre_type11(bottleneck_filters[0])
    input1 = torch.ones(4, 8, 30000)
    input2 = torch.ones(4, 8, 30000)
    output1 = testmodel1(input1, input2)
    print(output1.size())
    testmodel1.cuda()
    summary(testmodel1, [(8, 60000), (8, 60000)])

    # testmodel2 = multimodal_attention_layer(bottleneck_filters[0], downsize, two_stream=False)
    # output2 = testmodel2(input1, None)
    # print(output2.size())
    # testmodel2.cuda()
    # summary(testmodel2, [(8, 60000), (1, 1)])

    # testmodel3 = multimodal_attention_layer_type2(bottleneck_filters[0], downsize, two_stream=True)
    # input1 = torch.ones(4, 8, 60000)
    # input2 = torch.ones(4, 8, 60000)
    # output1 = testmodel3(input1, input2)
    # print(output1.size())
    #
    # testmodel4 = multimodal_attention_layer_type2(bottleneck_filters[0], downsize, two_stream=False)
    # output1 = testmodel4(input1, None)
    # print(output1.size())

    # testmodel5 = only_modal_attention(bottleneck_filters[0], downsize, two_stream=True)
    # input1 = torch.ones(4, 8, 60000)
    # input2 = torch.ones(4, 8, 60000)
    # output1 = testmodel5(input1, input2)
    # print(output1.size())

    # testmodel6 = only_merge(bottleneck_filters[0], downsize, two_stream=True)
    # input1 = torch.ones(4, 8, 60000)
    # input2 = torch.ones(4, 8, 60000)
    # output1 = testmodel6(input1, input2)
    # print(output1.size())