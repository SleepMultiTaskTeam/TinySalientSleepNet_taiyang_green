from model.o_encoder import TestEncoder
#from model.o_decoder import TestDecoder
from model.oo_decoder import Test_o_Decoder
from model.o_segmentClassifier import segmentClassifier
from torch import nn
# from torchsummary import summary
from model.o_multimodal_attention import *
import torch

# todo: 该如何进行打乱，每个人要用不同的编辑器
class TinySalientSleepNet(nn.Module):
    def __init__(self,kernel_size, filters, pooling_size, stride, sleep_epoch_len, sequence_len, layer_num, bottleneck_filters):
        super(TinySalientSleepNet, self).__init__()
        self.encoder = TestEncoder(kernel_size, filters, pooling_size, stride, sleep_epoch_len, sequence_len, layer_num, bottleneck_filters)
        self.decoder = Test_o_Decoder(kernel_size, bottleneck_filters, pooling_size, stride, sleep_epoch_len, sequence_len, layer_num, bottleneck_filters)
        self.classifier = segmentClassifier(kernel_size, bottleneck_filters, sleep_epoch_len)
        self.mma = multimodal_attention_layer(bottleneck_filters[0], downsize=2, two_stream=False)
        # self.decoder = Test_o_Decoder(kernel_size, filters, pooling_size, stride, sleep_epoch_len, sequence_len, layer_num, bottleneck_filters)
        # self.classifier = segmentClassifier(kernel_size, filters, sleep_epoch_len)
        # self.mma = multimodal_attention_layer(filters[0], downsize=2, two_stream=False)

    def forward(self, input, x_yasa=None):
        e5, e4, e3, e2, e1 = self.encoder(input)
        x = self.decoder(e5, e4, e3, e2, e1)
        x = self.mma(x, None)
        x = self.classifier(x, x_yasa)
        return x

class twostream_TinySalientSleepNet(nn.Module):
    def __init__(self, kernel_size, filters, pooling_size, stride, sleep_epoch_len, sequence_len, layer_num, bottleneck_filters):
        super(twostream_TinySalientSleepNet, self).__init__()
        self.encoder1 = TestEncoder(kernel_size, filters, pooling_size, stride, sleep_epoch_len, sequence_len, layer_num,
                                   bottleneck_filters)
        self.encoder2 = TestEncoder(kernel_size, filters, pooling_size, stride, sleep_epoch_len, sequence_len, layer_num,
                                   bottleneck_filters)
        self.decoder1 = Test_o_Decoder(kernel_size, bottleneck_filters, pooling_size, stride, sleep_epoch_len,
                                      sequence_len, layer_num, bottleneck_filters)
        self.decoder2 = Test_o_Decoder(kernel_size, bottleneck_filters, pooling_size, stride, sleep_epoch_len,
                                      sequence_len, layer_num, bottleneck_filters)

        self.classifier = segmentClassifier(kernel_size, bottleneck_filters, sleep_epoch_len)
        # self.mma = multimodal_attention_layer_type0(bottleneck_filters[0], downsize=2, two_stream=True)
        # self.mma = only_merge(bottleneck_filters[0], downsize=2, two_stream=True)
        # self.mma = only_modal_attention(bottleneck_filters[0], downsize=2, two_stream=True)
        # self.mma = multimodal_attention_layer_type2(bottleneck_filters[0], downsize=2, two_stream=True)
        # self.mma = multimodal_attention_layer_type4(bottleneck_filters[0])
        # self.mma = multimodal_megre_type00(bottleneck_filters[0])
        self.mma = multimodal_megre_v9_type9(bottleneck_filters[0])


    def forward(self, s1, s2):
        e5, e4, e3, e2, e1 = self.encoder1(s1)
        x1 = self.decoder1(e5, e4, e3, e2, e1)

        ee5, ee4, ee3, ee2, ee1 = self.encoder2(s2)
        x2 = self.decoder2(ee5, ee4, ee3, ee2, ee1)

        x = self.mma(x1, x2)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    # 参数
    kernel_size = 5
    filters = [16, 32, 32, 64, 128]
    # pooling_size = [10, 8, 6, 5]
    pooling_size = [10, 5, 5, 5]
    sleep_epoch_len = 3000
    sequence_len = 10
    stride = 1
    layer_num = 5
    bottleneck_filters = [16, 16, 32, 64, 128]
    # 构建模型与测试
    testmodel1 = TinySalientSleepNet(kernel_size, filters, pooling_size, stride, sleep_epoch_len,
                            sequence_len, layer_num, bottleneck_filters)
    inputs1 = torch.ones((2, 1, 37500 * 10))
    outputs = testmodel1(inputs1)
    print(outputs.shape)
    testmodel1.cuda()
    summary(testmodel1, (1, 60000))

    # 测试双流模型
    # testmodel2 = twostream_TinySalientSleepNet(kernel_size, filters, pooling_size, stride, sleep_epoch_len,
    #                                 sequence_len, layer_num, bottleneck_filters)
    # inputs2 = torch.ones((4, 1, 3000 * 10))
    # outputs = testmodel2(inputs1, inputs2)
    # print(outputs.shape)
    # testmodel2.cuda()
    # summary(testmodel2, [(1, 12000), (1, 12000)])