import torch
from torch import nn
from model.o_encoder import Encoder_Layer, multi_layer_transformer
# from torchsummary import summary

# class Decoder_Layer(nn.Module):
# 	def __init__(self, in_c, out_c, pool_s, kernel_size, stride):
# 		super(Decoder_Layer, self).__init__()
# 		self.model = nn.Sequential(
# 			nn.Conv1d(in_c, out_c, kernel_size, stride, kernel_size // 2),
# 			nn.ReLU(),
# 			nn.BatchNorm1d(out_c),
# 			nn.MaxPool1d(pool_s)
# 		)
#
# 	def forward(self, x):
# 		return self.model(x)


class Test_o_Decoder(nn.Module):
	# 注意，filters用bottleneck_filters
	def __init__(self, kernel_size, filters, pooling_size, stride, sleep_epoch_len, sequence_len, layer_num, bottleneck_filters):
		super(Test_o_Decoder, self).__init__()
		input_sequence_points = sleep_epoch_len * sequence_len
		# 尺寸追踪
		size = []
		size.append(input_sequence_points)
		for i in range(1, layer_num):
			size.append(size[i-1]//pooling_size[i-1])
		# 因为concat之后通道数翻倍，所以过滤器数错位一位，但是padding没有错位
		# 由于错位所以输出的时候正好通道数是filters[0]而不是输入时候的“通道数为1”
		# Decoder is same as Encoder
		# todo: 与以前的改变，比如以前是bottleneck是5和8，拼起来13，还要卷积到128的原filter大小，现在比如是32和16，卷完设定为16，相当于以bottleneck为过滤器标准了，过滤器小了一倍
		# todo: 原版的MFE参数是反向的，深层channel反而少
		# self.layer4 = Encoder_Layer(bottleneck_filters[4] + bottleneck_filters[3], filters[3], kernel_size, stride, padding=(size[3] * (stride - 1) - stride + kernel_size)//2)
		# self.layer3 = Encoder_Layer(filters[3] + bottleneck_filters[2], filters[2], kernel_size, stride, padding=(size[2] * (stride - 1) - stride + kernel_size) // 2)
		# self.layer2 = Encoder_Layer(filters[2] + bottleneck_filters[1], filters[1], kernel_size, stride, padding=(size[1] * (stride - 1) - stride + kernel_size) // 2)
		# self.layer1 = Encoder_Layer(filters[1] + bottleneck_filters[0], filters[0], kernel_size, stride, padding=(size[0] * (stride - 1) - stride + kernel_size) // 2)
		self.layer4 = Encoder_Layer(bottleneck_filters[4] + bottleneck_filters[3], filters[3], kernel_size, stride, padding=kernel_size//2)
		self.layer3 = Encoder_Layer(filters[3] + bottleneck_filters[2], filters[2], kernel_size, stride, padding=kernel_size//2)
		self.layer2 = Encoder_Layer(filters[2] + bottleneck_filters[1], filters[1], kernel_size, stride, padding=kernel_size//2)
		self.layer1 = Encoder_Layer(filters[1] + bottleneck_filters[0], filters[0], kernel_size, stride, padding=kernel_size//2)
		self.up4 = nn.ConvTranspose1d(bottleneck_filters[4], bottleneck_filters[4], kernel_size=pooling_size[3], stride=pooling_size[3])
		self.up3 = nn.ConvTranspose1d(filters[3], filters[3], kernel_size=pooling_size[2], stride=pooling_size[2])
		self.up2 = nn.ConvTranspose1d(filters[2], filters[2], kernel_size=pooling_size[1], stride=pooling_size[1])
		self.up1 = nn.ConvTranspose1d(filters[1], filters[1], kernel_size=pooling_size[0], stride=pooling_size[0])
		# self.trans4 = multi_layer_transformer(dim=filters[3], input_resolution=75, num_heads=8, layerNum=6,
		#                                       window_size=25, shift_size=12)
		# self.trans3 = multi_layer_transformer(dim=filters[2], input_resolution=375, num_heads=4, layerNum=2,
		#                                       window_size=125, shift_size=62)


	def forward(self, e5, e4, e3, e2, e1):
		# layer4
		x = self.up4(e5)
		x = torch.cat((x, e4), 1)
		x = self.layer4(x)
		# x = self.trans4(x.transpose(-1, -2)).transpose(-1, -2)
		# layer3
		x = self.up3(x)
		x = torch.cat((x, e3), 1)
		x = self.layer3(x)
		# x = self.trans3(x.transpose(-1, -2)).transpose(-1, -2)
		# layer2
		x = self.up2(x)
		x = torch.cat((x, e2), 1)
		x = self.layer2(x)
		# layer1
		x = self.up1(x)
		x = torch.cat((x, e1), 1)
		x = self.layer1(x)

		return x


if __name__ == '__main__':
	# 参数
	kernel_size = 5
	filters = [4, 8, 16, 32, 64]
	pooling_size = [10, 8, 5, 5]
	sleep_epoch_len = 3000
	sequence_len = 20
	stride = 1
	layer_num = 5
	bottleneck_filters = [i//2 for i in filters]
	# [2, 4, 8, 16, 32]


	#创建模型与测试
	testmodel = Test_o_Decoder(kernel_size, filters, pooling_size, stride, sleep_epoch_len,
							sequence_len, layer_num, bottleneck_filters)
	e1 = torch.ones((4, bottleneck_filters[0], 30000))
	e2 = torch.ones((4, bottleneck_filters[1], 3000))
	e3 = torch.ones((4, bottleneck_filters[2], 375))
	e4 = torch.ones((4, bottleneck_filters[3], 75))
	e5 = torch.ones((4, bottleneck_filters[4], 15))

	output = testmodel(e5, e4, e3, e2, e1)
	print(output.shape)

	testmodel.cuda()
	# summary(testmodel, [(bottleneck_filters[4], 20), (bottleneck_filters[3], 60),
	# 					(bottleneck_filters[2], 600), (bottleneck_filters[1], 6000),
	# 					(bottleneck_filters[0], 60000)])

