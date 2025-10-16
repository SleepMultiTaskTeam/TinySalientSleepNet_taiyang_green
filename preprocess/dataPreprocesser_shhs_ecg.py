import os
import random

import neurokit2 as nk
import numpy as np
from xml.etree import ElementTree as ET
import numpy as np
import torch
from scipy import interpolate
from my_dataset import MyDatasetThreePre, MyDataset, MyDatasetOnePre
from preprocess.loader import load_shhs_xml_stage_label, load_shhs_edf



class DataPreprocesserSHHS:
	def __init__(self, data_dir, p_names, input_epoch_num, sampling_rate, fold_num):
		self.fold_num = fold_num

		data = []
		label = []
		for name in p_names:
			stage_label = load_shhs_xml_stage_label(os.path.join(data_dir, name + "-nsrr.xml"))
			raw_dataframe = load_shhs_edf(os.path.join(data_dir, name + ".edf"))

			# 提取RRI，去除时间维度
			xall = raw_dataframe.values[:]
			xall = xall.swapaxes(0, 1)
			xall = xall[1:]

			# 分通道
			raw_ecg = xall[0]
			raw_thor = xall[1]
			raw_abdo = xall[2]

			# 整理ecg
			# ecg_cleaned = nk.ecg_clean(raw_ecg, sampling_rate=sampling_rate)

			# 提取峰，提取RRI
			_, info = nk.ecg_peaks(xall[0], sampling_rate=sampling_rate)
			peaks = info["ECG_R_Peaks"]
			# Convert peaks to intervals
			rri = np.diff(peaks) / sampling_rate * 1000  # 毫秒单位的RRI
			rri_time = peaks[1:]  # 以采样点为单位的时间轴，RRI会舍弃第一个峰，从第二个峰开始计算
			# rri_real_time = np.array(peaks[1:]) / sampling_rate

			# 插值
			f = interpolate.interp1d(rri_time, rri, kind="linear")  # 生成插值模板
			new_rri_time = np.linspace(rri_time[0], rri_time[-1], 1 + rri_time[-1] - rri_time[0])  # 生成插值的时间点
			# new_rri_time = np.arange(rri_time[0], rri_time[-1], sampling_rate*0.2)  # 5Hz的RRI，是原长度的1/25（因为原序列125Hz）
			new_rri = f(new_rri_time)
			new_rri_real_time = new_rri_time / sampling_rate

			# 由于RRI和插值的错位，第一个睡眠期和最后一个睡眠期整个裁剪掉
			# 剔除前后30min，前和后因为之前提取RRI都多剃掉了一些，所以这些取消掉来对齐。
			# 已核验，剔除正确
			left_cut = 30 * 60 * sampling_rate - rri_time[0]
			right_cut = - (30 * 60 * sampling_rate - (len(raw_ecg) - rri_time[-1] - 1))
			new_rri = new_rri[left_cut:right_cut]
			# new_rri_time = new_rri_time[left_cut:right_cut]
			# new_rri_real_time = new_rri_real_time[left_cut:right_cut]

			# 睡眠期剔除前后30min，前和后分别是30*60/30 = 60个epoch
			stage_label = stage_label[60:-60]

			# 另外两个信号也剔除前后30min
			left_cut = 30 * 60 * sampling_rate
			right_cut = - (30 * 60 * sampling_rate)
			new_thor = raw_thor[left_cut:right_cut]
			new_abdo = raw_abdo[left_cut:right_cut]

			# 结合，如果之前RRI剔除前后30min的计算有误，这里是不能合起来的，尺寸不对
			new_xall = np.concatenate([new_rri[np.newaxis, :], new_thor[np.newaxis, :], new_abdo[np.newaxis, :]],
			                          axis=0)
			new_xall = new_xall.swapaxes(0, 1)  # 时间轴,3通道


			p = 0
			x_group = None
			y_group = None
			while p < len(stage_label):
				if p + input_epoch_num < len(stage_label):
					x = new_xall[p * 30 * sampling_rate: (p + input_epoch_num) * 30 * sampling_rate]
					y = stage_label[p: p + input_epoch_num]

					x = x.swapaxes(0, 1)
					# x = x[0:1]  # 在这行可以选择只保留心电通道
					x = x[1:2]  # 在这行可以选择只保留胸带通道
					# x = x[2:3]  # 在这行可以选择只保留心电通道

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

		c = list(zip(x, y))
		random.shuffle(c)
		x[:], y[:] = zip(*c)

		self.x = x
		self.y = y

		# # 进行最大最小归一化
		# scaler = MinMaxScaler()  # 实例化
		# scaler.fit(x[0])  # 在这里本质是生成min(x)和max(x)
		# x[0] = scaler.transform(x[0])  # 通过接口导出结果
		# scaler.fit(x[1])  # 在这里本质是生成min(x)和max(x)
		# x[1] = scaler.transform(x[1])  # 通过接口导出结果

		# shuffle
		# index = [i3 for i3 in range(len(self.x))]
		# np.random.shuffle(index)
		# self.x = self.x[index]
		# self.y = self.y[index]

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
