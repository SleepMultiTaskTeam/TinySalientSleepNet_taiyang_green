import torch
from torch import nn
# from torchsummary import summary
from model.mysummary import summary
from model.swin_transformer_1d import SwinTransformerBlock
import torch.nn.functional as F

# todo: 只用了一个卷积，原版一层是2个卷积
class Encoder_Layer(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, padding):
        super(Encoder_Layer, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size, stride, padding),
            nn.ReLU(),
            nn.BatchNorm1d(out_c),
            # nn.Conv1d(out_c, out_c, kernel_size, stride, padding),
            # nn.ReLU(),
            # nn.BatchNorm1d(out_c),
            nn.Conv1d(out_c, out_c, kernel_size, stride, padding),
            nn.ReLU(),
            nn.BatchNorm1d(out_c)
            # nn.Dropout(p=0.2)
        )

    def forward(self, x):
        return self.model(x)

class down_5_PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
# 用sequence_size替代input_resolution
    def __init__(self, sequence_size, dim, down_size = 5, norm_layer=nn.LayerNorm, channelup_size = 2):
        super().__init__()
        self.sequence_size = sequence_size
        self.dim = dim
        self.down_size = down_size
        # reduction出去的维度就是我们想要的维度
        self.reduction = nn.Linear(down_size * dim, channelup_size * dim, bias=False)
        self.norm = norm_layer(down_size * dim)
        # self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        # self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # H = self.sequence_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        # x = x.view(B, H, W, C) #不需要扩展维度，我们本来就是1维的

        # 它是用4个不同的起始点加步幅采集一个窗内不同的角
        # x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        # x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        # x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        # x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        # x0 = x[:, 0::5, :]
        # x1 = x[:, 1::5, :]
        # x2 = x[:, 2::5, :]
        # x3 = x[:, 3::5, :]
        # x4 = x[:, 4::5, :]
        after_split = []
        for i in range(self.down_size):
            after_split.append(x[:, i::self.down_size, :])
        # 然后把4个角（这里是2个角）拼起来
        # 本来4个拼起来通道数会多4倍，我这里只能多2倍
        # 其实我可以设置成任意倍率（就是控制拼接点的个数呗），但是60000除以4最多除以2次，然后就会出分数，所以降采样2倍比较好调
        # x = torch.cat([x0, x1, x2, x3, x4], -1)  # B H/5 5*C
        # x = x.view(B, -1, 5 * C)  # B H/5 5*C

        x = torch.cat(after_split, -1)  # B H/5 5*C
        x = x.view(B, -1, self.down_size * C)  # B H/5 5*C
        # linear只会操作最后一个维度
        x = self.norm(x)
        x = self.reduction(x)

        return x

# class Encoder_Layer(nn.Module):
#     def __init__(self, in_c, out_c, kernel_size, stride, padding):
#         super(Encoder_Layer, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv1d(in_c, out_c, kernel_size, stride, padding),
#             nn.ReLU(),
#             nn.BatchNorm1d(out_c),
#             nn.Conv1d(out_c, out_c, kernel_size, stride, padding),
#             nn.ReLU(),
#             nn.BatchNorm1d(out_c),
#         )
#
#     def forward(self, x):
#         return self.model(x)

class MSE(nn.Module):
    def __init__(self, kernel_size, in_filter, out_filter):
        super(MSE, self).__init__()
        self.dConv = nn.ModuleList()
        # 5+ (5-1) * (dilation-1) = (k-1)*(dilation-1) + k
        for i in range(4):
            conv = nn.Sequential(
                nn.Conv1d(in_filter, out_filter, kernel_size, padding=((i + 1) * (kernel_size) - i)// 2, dilation=i+1),
                nn.ReLU(),
                nn.BatchNorm1d(out_filter)
            )
            self.dConv.append(conv)

        self.Down1 = nn.Sequential(
            nn.Conv1d(out_filter * 4, out_filter * 2, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(out_filter * 2)
        )

        self.Down2 = nn.Sequential(
            nn.Conv1d(out_filter * 2, out_filter, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(out_filter)
        )
        # self.c2 = nn.Conv1d(in_filter, out_filter, kernel_size, padding=kernel_size // 2, dilation=2)
        # self.c3 = nn.Conv1d(in_filter, out_filter, kernel_size, padding=kernel_size // 2, dilation=3)
        # self.c4 = nn.Conv1d(in_filter, out_filter, kernel_size, padding=kernel_size // 2, dilation=4)
    def forward(self, x):
        x1 = self.dConv[0](x)
        x2 = self.dConv[1](x)
        x3 = self.dConv[2](x)
        x4 = self.dConv[3](x)

        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.Down1(x)
        x = self.Down2(x)

        return x

class multi_layer_transformer(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, layerNum, window_size=25, shift_size=12):
        super(multi_layer_transformer, self).__init__()

        self.layerNum = layerNum
        self.multi_trans = nn.ModuleList()

        for i in range(layerNum//2):
            trans1 = SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size)
            trans2 = SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size,
                                          shift_size=shift_size)
            self.multi_trans.append(trans1)
            self.multi_trans.append(trans2)

    def forward(self, x):
        for i in range(self.layerNum):
            x = self.multi_trans[i](x)

        return x

class MS_transformer(nn.Module):
    def __init__(self, kernel_size, in_filter, out_filter, input_resolution, num_heads, window_size, shift_size):
        super(MS_transformer, self).__init__()
        self.dConv = nn.ModuleList()
        self.trans = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.in_filter = in_filter
        # 5+ (5-1) * (dilation-1) = (k-1)*(dilation-1) + k
        # layerNum_of_transfomer = (6, 4, 2, 2)
        # layerNum_of_transfomer = (2, 2, 2, 2)
        # layerNum_of_transfomer = (6, 6, 4, 2)
        # layerNum_of_transfomer = (12, 2, 2, 2)
        # layerNum_of_transfomer = (6, 0, 0, 0)
        layerNum_of_transfomer = (6, 6, 6, 6)
        for i in range(4):
            conv = nn.Sequential(
                nn.Conv1d(in_filter, out_filter, kernel_size, padding=((i + 1) * (kernel_size) - i)// 2, dilation=i+1),
                nn.ReLU(),
                nn.BatchNorm1d(out_filter)
            )
            self.dConv.append(conv)

            trans = nn.Sequential(
                multi_layer_transformer(out_filter, input_resolution, num_heads, layerNum_of_transfomer[i],
                                        window_size, shift_size),
            )
            self.trans.append(trans)

            self.norm.append(nn.BatchNorm1d(out_filter))

        self.Down1 = nn.Sequential(
            nn.Conv1d(out_filter * 4, out_filter * 2, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(out_filter * 2)
        )

        self.Down2 = nn.Sequential(
            nn.Conv1d(out_filter * 2, out_filter, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(out_filter)
        )

        # self.bottleneck = nn.Sequential(
        #     nn.Conv1d(out_filter * 4, out_filter, 1),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(out_filter)
        # )

        self.avgPool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(4 * out_filter, 2 * out_filter, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(2 * out_filter, 4 * out_filter, bias=False),
            nn.Sigmoid()
        )
        # self.c2 = nn.Conv1d(in_filter, out_filter, kernel_size, padding=kernel_size // 2, dilation=2)
        # self.c3 = nn.Conv1d(in_filter, out_filter, kernel_size, padding=kernel_size // 2, dilation=3)
        # self.c4 = nn.Conv1d(in_filter, out_filter, kernel_size, padding=kernel_size // 2, dilation=4)
    def forward(self, x):
        x1 = self.dConv[0](x)
        x1 = self.trans[0](x1.transpose(-1,-2)).transpose(-1,-2)
        x1 = self.norm[0](x1)

        x2 = self.dConv[1](x)
        x2 = self.trans[1](x2.transpose(-1, -2)).transpose(-1, -2)
        x2 = self.norm[1](x2)

        x3 = self.dConv[2](x)
        x3 = self.trans[2](x3.transpose(-1, -2)).transpose(-1, -2)
        x3 = self.norm[2](x3)

        x4 = self.dConv[3](x)
        x4 = self.trans[3](x4.transpose(-1, -2)).transpose(-1, -2)
        x4 = self.norm[3](x4)


        x = torch.cat([x1, x2, x3, x4], dim=1)

        b, c, _ = x.size()
        y = self.avgPool(x).view(b, c)  # 去掉最后一维，相当于通道展平的步骤
        y = self.fc(y).view(b, c, 1)  # 还原维度
        y = y.expand_as(x)
        x = x * y

        x = self.Down1(x)
        x = self.Down2(x)
        # x = self.bottleneck(x)

        return x

class MSGE(nn.Module):
    def __init__(self, kernel_size, in_filter, out_filter, input_resolution, num_heads, window_size, shift_size, transNum):
        super(MSGE, self).__init__()
        self.dConv = nn.ModuleList()
        # 5+ (5-1) * (dilation-1) = (k-1)*(dilation-1) + k
        for i in range(4):
            conv = nn.Sequential(
                nn.Conv1d(in_filter, out_filter, kernel_size, padding=((i + 1) * (kernel_size) - i)// 2, dilation=i+1),
                nn.ReLU(),
                nn.BatchNorm1d(out_filter)
            )
            self.dConv.append(conv)

        self.trans = multi_layer_transformer(in_filter, input_resolution, num_heads, transNum,
                                window_size, shift_size)
        self.downTrans = nn.Sequential(
            nn.Conv1d(in_filter, out_filter, kernel_size=1),
            nn.BatchNorm1d(out_filter)
        )

        self.Down1 = nn.Sequential(
            nn.Conv1d(out_filter * 5, out_filter * 3, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(out_filter * 3)
        )

        self.Down2 = nn.Sequential(
            nn.Conv1d(out_filter * 3, out_filter, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(out_filter)
        )
        # self.c2 = nn.Conv1d(in_filter, out_filter, kernel_size, padding=kernel_size // 2, dilation=2)
        # self.c3 = nn.Conv1d(in_filter, out_filter, kernel_size, padding=kernel_size // 2, dilation=3)
        # self.c4 = nn.Conv1d(in_filter, out_filter, kernel_size, padding=kernel_size // 2, dilation=4)
    def forward(self, x):
        x1 = self.dConv[0](x)
        x2 = self.dConv[1](x)
        x3 = self.dConv[2](x)
        x4 = self.dConv[3](x)
        xg = self.trans(x.transpose(-1,-2)).transpose(-1,-2)
        xg = self.downTrans(xg)

        x = torch.cat([x1, x2, x3, x4, xg], dim=1)

        x = self.Down1(x)
        x = self.Down2(x)

        return x

class TestEncoder(nn.Module):
    def __init__(self,kernel_size, filters, pooling_size, stride, sleep_epoch_len, sequence_len, layer_num, bottleneck_filters):
        super(TestEncoder, self).__init__()
        # 追踪每层的输出尺寸
        input_sequence_points = sleep_epoch_len * sequence_len
        size = []
        size.append(input_sequence_points)
        for i in range(1, layer_num):
            size.append(size[i - 1] // pooling_size[i - 1])
        # 第五层只有个池化，所以第五层的尺寸这里用不到
        # 长 = 宽 = H‘=[(H - k_h + 2p) / s + 1]
        # (H' - 1)s = H - k_h + 2p
        # 2p = sH' - s - H + k_h
        # 设H' = H
        # 2p = (s-1)H - s + k_h
        # Stride = 3, H =

        # Conv
        # 3通道 250604 by zmj
        # 4通道 250627 by zmj
        # 3通道 250707 by MJZ
        self.layer1 = Encoder_Layer(3, filters[0], kernel_size, stride,
                                    (size[0] * (stride - 1) - stride + kernel_size) // 2)
        self.layer2 = Encoder_Layer(filters[0], filters[1], kernel_size, stride,
                                    (size[1] * (stride - 1) - stride + kernel_size) // 2)
        self.layer3 = Encoder_Layer(filters[1], filters[2], kernel_size, stride,
                                    (size[2] * (stride - 1) - stride + kernel_size) // 2)
        self.layer4 = Encoder_Layer(filters[2], filters[3], kernel_size, stride,
                                    (size[3] * (stride - 1) - stride + kernel_size) // 2)
        self.layer5 = Encoder_Layer(filters[3], filters[4], kernel_size, stride,
                                    (size[4] * (stride - 1) - stride + kernel_size) // 2)

        # swinTrans
        # channel 4, 8, 16, 32, 64
        # 60000, 6000, 750, 125, 25
        # 60000, 6000, 750, 150, 30
        # pool 10,8,5,5
        # 30000, 3000, 375, 75, 15
        # self.trans2_1 = SwinTransformerBlock(dim=filters[1], input_resolution=6000, num_heads=4, window_size=500)
        # self.trans2_2 = SwinTransformerBlock(dim=filters[1], input_resolution=6000, num_heads=4, window_size=500, shift_size=250)

        # self.trans3_1 = SwinTransformerBlock(dim=filters[2], input_resolution=375, num_heads=4, window_size=25)
        # self.trans3_2 = SwinTransformerBlock(dim=filters[2], input_resolution=375, num_heads=4, window_size=25, shift_size=12)



        # self.trans4_1 = SwinTransformerBlock(dim=filters[3], input_resolution=75, num_heads=8, window_size=25)
        # self.trans4_2 = SwinTransformerBlock(dim=filters[3], input_resolution=75, num_heads=8, window_size=25,
        #                                      shift_size=12)
        # self.trans4_3 = SwinTransformerBlock(dim=filters[3], input_resolution=75, num_heads=8, window_size=25)
        # self.trans4_4 = SwinTransformerBlock(dim=filters[3], input_resolution=75, num_heads=8, window_size=25,
        #                                      shift_size=12)
        # self.trans4_5 = SwinTransformerBlock(dim=filters[3], input_resolution=75, num_heads=8, window_size=25)
        # self.trans4_6 = SwinTransformerBlock(dim=filters[3], input_resolution=75, num_heads=8, window_size=25,
        #                                      shift_size=12)

        # self.trans5_1 = SwinTransformerBlock(dim=filters[4], input_resolution=15, num_heads=16, window_size=25)
        # self.trans5_2 = SwinTransformerBlock(dim=filters[4], input_resolution=15, num_heads=16, window_size=25,
        #                                      shift_size=12)
        # self.trans5_3 = SwinTransformerBlock(dim=filters[4], input_resolution=25, num_heads=16, window_size=25)
        # self.trans5_4 = SwinTransformerBlock(dim=filters[4], input_resolution=25, num_heads=16, window_size=25,
        #                                      shift_size=12)
        # self.trans5_5 = SwinTransformerBlock(dim=filters[4], input_resolution=25, num_heads=16, window_size=25)
        # self.trans5_6 = SwinTransformerBlock(dim=filters[4], input_resolution=25, num_heads=16, window_size=25,
        #                                      shift_size=12)

        # self.trans2 = multi_layer_transformer(dim=filters[1], input_resolution=3000, num_heads=4, layerNum=6, window_size=25,
        #                                       shift_size=12)
        self.trans3 = multi_layer_transformer(dim=filters[2], input_resolution=375, num_heads=4, layerNum=2, window_size=25,
                                              shift_size=12)
        self.trans4 = multi_layer_transformer(dim=filters[3], input_resolution=75, num_heads=8, layerNum=6, window_size=25,
                                              shift_size=12)
        self.trans5 = multi_layer_transformer(dim=filters[4], input_resolution=15, num_heads=16, layerNum=2, window_size=25,
                                              shift_size=12)

        # self.down2 = down_5_PatchMerging(sequence_size=3000, dim=64, down_size=8)
        # self.down3 = down_5_PatchMerging(sequence_size=375, dim=128, down_size=5)
        # self.down4 = down_5_PatchMerging(sequence_size=75, dim=256, down_size=5)


        # channel 4, 8, 16, 32, 64
        # 12000, 1200, 150, 25, 5
        # self.trans2_1 = SwinTransformerBlock(dim=filters[1], input_resolution=1200, num_heads=4, window_size=25)
        # self.trans2_2 = SwinTransformerBlock(dim=filters[1], input_resolution=1200, num_heads=4, window_size=25, shift_size=12)

        # self.trans3_1 = SwinTransformerBlock(dim=filters[2], input_resolution=150, num_heads=4, window_size=25)
        # self.trans3_2 = SwinTransformerBlock(dim=filters[2], input_resolution=150, num_heads=4, window_size=25, shift_size=12)
        #
        # self.trans4_1 = SwinTransformerBlock(dim=filters[3], input_resolution=25, num_heads=8, window_size=25)
        # self.trans4_2 = SwinTransformerBlock(dim=filters[3], input_resolution=25, num_heads=8, window_size=25,
        #                                      shift_size=12)
        # self.trans4_3 = SwinTransformerBlock(dim=filters[3], input_resolution=25, num_heads=8, window_size=25)
        # self.trans4_4 = SwinTransformerBlock(dim=filters[3], input_resolution=25, num_heads=8, window_size=25,
        #                                      shift_size=12)
        # self.trans4_5 = SwinTransformerBlock(dim=filters[3], input_resolution=25, num_heads=8, window_size=25)
        # self.trans4_6 = SwinTransformerBlock(dim=filters[3], input_resolution=25, num_heads=8, window_size=25,
        #                                      shift_size=12)
        #
        # self.trans5_1 = SwinTransformerBlock(dim=filters[4], input_resolution=5, num_heads=16, window_size=5)
        # self.trans5_2 = SwinTransformerBlock(dim=filters[4], input_resolution=5, num_heads=16, window_size=5,
        #                                      shift_size=5)

        # MaxPool
        self.p1 = nn.MaxPool1d(pooling_size[0])
        self.p2 = nn.MaxPool1d(pooling_size[1])
        self.p3 = nn.MaxPool1d(pooling_size[2])
        self.p4 = nn.MaxPool1d(pooling_size[3])

        # bottleneck
        # 卷积核为1，步幅为1的卷积层作为bottleneck
        # self.bottleneck1 = nn.Conv1d(filters[0], bottleneck_filters[0], 1, 1)
        # self.bottleneck2 = nn.Conv1d(filters[1], bottleneck_filters[1], 1, 1)
        self.bottleneck3 = nn.Conv1d(filters[2], bottleneck_filters[2], 1, 1)
        self.bottleneck4 = nn.Conv1d(filters[3], bottleneck_filters[3], 1, 1)
        self.bottleneck5 = nn.Conv1d(filters[4], bottleneck_filters[4], 1, 1)

        self.MSE1 = MSE(kernel_size, filters[0], bottleneck_filters[0])
        self.MSE2 = MSE(kernel_size, filters[1], bottleneck_filters[1])
        # self.MS_trans3 = MS_transformer(kernel_size, filters[2], bottleneck_filters[2], input_resolution=375,
        #                                 num_heads=4, window_size=25, shift_size=12)
        # self.MS_trans4 = MS_transformer(kernel_size, filters[3], bottleneck_filters[3], input_resolution=75,
        #                                 num_heads=8, window_size=25, shift_size=12)
        # self.MS_trans5 = MS_transformer(kernel_size, filters[4], bottleneck_filters[4], input_resolution=15,
        #                                 num_heads=16, window_size=25, shift_size=12)
        # self.MSGE3 = MSGE(kernel_size, filters[2], bottleneck_filters[2], input_resolution=150,
        #                                  num_heads=4, window_size=25, shift_size=12, transNum=2)
        # self.MSGE4 = MSGE(kernel_size, filters[3], bottleneck_filters[3], input_resolution=25,
        #                   num_heads=8, window_size=25, shift_size=12, transNum=6)
        # self.MSGE5 = MSGE(kernel_size, filters[4], bottleneck_filters[4], input_resolution=5,
        #                   num_heads=16, window_size=25, shift_size=12, transNum=2)

    # self.layers = []
    # layer_num >= 2
    # self.layers.append(Encoder_Layer(1, filters[0], pooling_size[0], kernel_size))
    # for i in range(1, layer_num):
    # 	self.layers.append(Encoder_Layer(filters[i - 1], filters[i], pooling_size[i], kernel_size))

    def forward(self, x):
        # layer_num >= 2
        # output1 = self.layer1(x)
        # output2 = self.layer2(F.max_pool1d(output1, self.pooling_size[0]))
        # output3 = self.layer3(F.max_pool1d(output2, self.pooling_size[1]))
        # output4 = self.layer4(F.max_pool1d(output3, self.pooling_size[2]))
        # output5 = F.max_pool1d(output4, self.pooling_size[3])
        out1 = self.layer1(x)

        out2 = self.p1(out1)
        out2 = self.layer2(out2)
        # out2 = self.trans2(out2.transpose(-1, -2)).transpose(-1, -2)

        # out2 = self.trans2_1(out2.transpose(-1,-2))
        # out2 = self.trans2_2(out2).transpose(-1,-2)

        out3 = self.p2(out2)
        out3 = self.layer3(out3)
        # out3 = self.down2(out2.transpose(-1,-2)).transpose(-1,-2)
        # out3 = self.trans3(out3.transpose(-1,-2)).transpose(-1,-2)

        # out3 = self.trans3_1(out3.transpose(-1,-2))
        # out3 = self.trans3_2(out3).transpose(-1,-2)

        out4 = self.p3(out3)
        out4 = self.layer4(out4)
        # out4 = self.down3(out3.transpose(-1,-2)).transpose(-1,-2)
        # out4 = self.trans4(out4.transpose(-1, -2)).transpose(-1, -2)

        # out4 = self.trans4_1(out4.transpose(-1,-2))
        # out4 = self.trans4_2(out4)
        # out4 = self.trans4_3(out4)
        # out4 = self.trans4_4(out4)
        # out4 = self.trans4_5(out4)
        # out4 = self.trans4_6(out4).transpose(-1,-2)

        out5 = self.p4(out4)
        out5 = self.layer5(out5)
        # out5 = self.down4(out4.transpose(-1,-2)).transpose(-1,-2)
        # out5 = self.trans5(out5.transpose(-1, -2)).transpose(-1, -2)

        # out5 = self.trans5_1(out5.transpose(-1,-2))
        # out5 = self.trans5_2(out5).transpose(-1,-2)

        # out5 = self.trans5_3(out5)
        # out5 = self.trans5_4(out5)
        # out5 = self.trans5_5(out5)
        # out5 = self.trans5_6(out5)

        # out1 = self.bottleneck1(out1)
        # out2 = self.bottleneck2(out2)
        out1 = self.MSE1(out1)
        out2 = self.MSE2(out2)
        # out2 = self.bottleneck2(out2)
        out3 = self.bottleneck3(out3)
        out4 = self.bottleneck4(out4)
        out5 = self.bottleneck5(out5)
        # out3 = self.MS_trans3(out3)
        # out4 = self.MS_trans4(out4)
        # out5 = self.MS_trans5(out5)
        # out3 = self.MSGE3(out3)
        # out4 = self.MSGE4(out4)
        # out5 = self.MSGE5(out5)

        return out5, out4, out3, out2, out1

    def getLayer(self):
        return self.layer1, self.layer2, self.layer3, self.layer4, self.layer5





if __name__ == '__main__':
    # 想要参数全部一步到位可见那么就得创建集
    # todo:stride必须是奇数（如果根据公式每层size都是偶数，卷积核是奇数的话）
    # 参数
    kernel_size = 5
    filters = [32, 64, 128, 256, 512]
    #pooling_size = [10, 8, 6, 5]
    pooling_size = [10, 8, 5, 5]
    sleep_epoch_len = 3000
    sequence_len = 10
    stride = 1
    layer_num = 5
    bottleneck_filters = [16, 32, 64, 128, 256]

    # 创建模型与测试
    testmodel = TestEncoder(kernel_size, filters, pooling_size, stride, sleep_epoch_len,
                            sequence_len, layer_num, bottleneck_filters)
    inputs = torch.ones((4, 1, 3000 * 10))
    outputs = testmodel(inputs)
    print(outputs[0].shape)
    print(len(outputs))
    testmodel.cuda()
    summary(testmodel, (1, 30000))
    # device = torch.device('cuda:0')
    # testmodel.to(device)
    # testmodel.cuda()
    # device = torch.device('cpu')
    # testmodel.to(device)
    # print(list(testmodel.parameters()))
    # for model in testmodel.layers:
    # 	#所以cuda也得一个一个转！！
    testmodel.cuda()
    # summary(testmodel, (1, 12000))
    # todo: 注意，因为testmodel这个模态完全不是正常的网络状态，它的返回是列表，所以没有device等属性，转cuda什么的都得一对一转，parameters也只能挨个调用
    layers = testmodel.getLayer()
    print()

    downtest = down_5_PatchMerging(375, dim=32)
    inputs = torch.zeros((4, 32, 375))
    outputs = downtest(inputs.transpose(-1, -2)).transpose(-1, -2)
    print(outputs.shape)
