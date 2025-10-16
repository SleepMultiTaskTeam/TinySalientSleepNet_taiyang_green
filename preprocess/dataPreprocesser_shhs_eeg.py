import os
import random
import neurokit2 as nk
import numpy as np
from xml.etree import ElementTree as ET
import numpy as np
import torch
from my_dataset import MyDatasetThreePre, MyDataset, MyDatasetOnePre
from preprocess.loader import load_shhs_xml_stage_label, load_shhs_edf



class DataPreprocesserSHHS:
	def __init__(self, data_dir, p_names, input_epoch_num, freq, fold_num):
		self.fold_num = fold_num

		data = []
		label = []
		for name in p_names:
			stage_label = load_shhs_xml_stage_label(os.path.join(data_dir, name + "-nsrr.xml"))
			raw_dataframe = load_shhs_edf(os.path.join(data_dir, name + ".edf"))

			# # 提取RRI，去除时间维度
			# raw_data = raw_dataframe.swapaxes(0, 1)
			# raw_data = raw_data[1:]  # 去除时间信息
			# rri, _ = nk.ecg_peaks(raw_data[0], sampling_rate=freq)
			# rri = rri.values
			# raw_data[0] = rri.reshape(-1)
			# raw_data = raw_data.swapaxes(0, 1)

			p = 0
			x_group = None
			y_group = None
			while p < len(stage_label):
				if p + input_epoch_num < len(stage_label):
					x = raw_dataframe.values[p * 30 * freq: (p + input_epoch_num) * 30 * freq]
					y = stage_label[p: p + input_epoch_num]

					# 提取RRI，去除时间维度，交换channel位置
					x = x.swapaxes(0, 1)
					x = x[1:]
					# x = x[1:2]
					rri, _ = nk.ecg_peaks(x[0], sampling_rate=125, method="pantompkins1985")
					rri = rri.values
					x[0] = rri.reshape(-1)
				else:
					break  # 多出的一段不要了
				p += input_epoch_num

				if x_group is None:
					x_group = x[np.newaxis, :]
					y_group = y[np.newaxis, :]
				else:
					x_group = np.append(x_group, x[np.newaxis, :], axis=0)
					y_group = np.append(y_group, y[np.newaxis, :], axis=0)

			# # 新增按人分的维度
			# x_group = x_group[np.newaxis, :]
			# y_group = y_group[np.newaxis, :]

			# 转gpu
			x_group = torch.from_numpy(x_group).float().cuda()
			y_group = torch.from_numpy(y_group).float().cuda()

			data.append(x_group)
			label.append(y_group)
			# if len(data) == 0:
			# 	data = x_group
			# 	label = y_group
			# else:
			# 	data = np.append(data, x_group, axis=0)
			# 	label = np.append(label, y_group, axis=0)

		print("load data finish")

		x = data
		y = label

		# shuffle
		# c = list(zip(x, y))
		# random.shuffle(c)
		# x[:], y[:] = zip(*c)

		self.x = x
		self.y = y

		# # 进行最大最小归一化
		# scaler = MinMaxScaler()  # 实例化
		# scaler.fit(x[0])  # 在这里本质是生成min(x)和max(x)
		# x[0] = scaler.transform(x[0])  # 通过接口导出结果
		# scaler.fit(x[1])  # 在这里本质是生成min(x)和max(x)
		# x[1] = scaler.transform(x[1])  # 通过接口导出结果

		# shuffle
		# index = [i3 for i3 in range(x.shape[1])]
		# np.random.shuffle(index)
		# x[0] = x[0][index]
		# x[1] = x[1][index]
		# y = y[index]

		# switch channel
		# x = x.transpose(0, 1, 3, 2)

		# to device
		# self.x = torch.from_numpy(x).float().cuda()
		# self.y = torch.from_numpy(y).float().cuda()

	def gen_fold_data(self, fold, train_val_cut_rate):
		print(f"===============gen fold {fold} data===============")

		# one_fold_num
		one_fold_num = int(len(self.x) / self.fold_num)
		print("one_fold_num: ", one_fold_num)
		# 1折测试，剩下的给训练和验证，训练验证按cut_rate切分
		test_cut_begin = one_fold_num * fold
		test_cut_end = one_fold_num * (fold + 1)
		print("test_cut_begin: ", test_cut_begin, "test_cut_end: ", test_cut_end)

		# cut
		train_val_x = self.x[0:test_cut_begin] + self.x[test_cut_end:]
		val_cut_begin = int(len(train_val_x) * train_val_cut_rate)
		train_x = train_val_x[0:val_cut_begin]
		val_x = train_val_x[val_cut_begin:]
		test_x = self.x[test_cut_begin: test_cut_end]

		train_val_y = self.y[0:test_cut_begin] + self.y[test_cut_end:]
		train_y = train_val_y[0:val_cut_begin]
		val_y = train_val_y[val_cut_begin:]
		test_y = self.y[test_cut_begin: test_cut_end]

		# train_val_x = torch.cat((self.x[0:test_cut_begin], self.x[test_cut_end:]), dim=0)
		# val_cut_begin = int(len(train_val_x) * train_val_cut_rate)
		# train_x = train_val_x[0:val_cut_begin]
		# val_x = train_val_x[val_cut_begin:]
		# test_x = self.x[test_cut_begin: test_cut_end]
		#
		# train_val_y = torch.cat((self.y[0:test_cut_begin], self.y[test_cut_end:]), dim=0)
		# train_y = train_val_y[0:val_cut_begin]
		# val_y = train_val_y[val_cut_begin:]
		# test_y = self.y[test_cut_begin: test_cut_end]

		# 把人数展平
		train_x = torch.cat(train_x, dim=0)
		train_y = torch.cat(train_y, dim=0)
		val_x = torch.cat(val_x, dim=0)
		val_y = torch.cat(val_y, dim=0)
		test_x = torch.cat(test_x, dim=0)
		test_y = torch.cat(test_y, dim=0)

		# train_x = train_x.reshape(-1, train_x.shape[2], train_x.shape[3])
		# train_y = train_y.reshape(-1, train_y.shape[2])
		# val_x = val_x.reshape(-1, val_x.shape[2], val_x.shape[3])
		# val_y = val_y.reshape(-1, val_y.shape[2])
		# test_x = test_x.reshape(-1, test_x.shape[2], test_x.shape[3])
		# test_y = test_y.reshape(-1, test_y.shape[2])

		train_dataset = MyDataset(train_x, train_y)
		val_dataset = MyDataset(val_x, val_y)
		test_dataset = MyDataset(test_x, test_y)

		print("train data shape: ", len(train_x))
		print("val data shape: ", len(val_x))
		print("test data shape: ", len(test_x))

		return train_dataset, val_dataset, test_dataset
