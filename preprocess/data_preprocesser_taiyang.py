import os
import random
import numpy as np
from xml.etree import ElementTree as ET
import numpy as np
import mne
import torch
from scipy import interpolate
import scipy.io as sio
from scipy import signal
import json
import pickle as pkl
from my_dataset import *
import xml.etree.ElementTree as ET
import pandas as pd
import logging
from datetime import timedelta

logger = logging.getLogger('test_run')

class DataPreprocesser:
	def __init__(self, datasets_dir, channels_selected, dataset, input_epoch_num, sampling_rate, label_available, args):
		self.input_epoch_num = input_epoch_num
		self.label_available = label_available
		self.window_num_by_sub = []
		self.last_epoch_num_by_sub = []
		taiyang_data=[]
		taiyang_label=[]
		self.sleep_times = []
		self.start_times = []
		self.end_times = []
		if dataset=="taiyang":
			# 读取仅做测试，4通道/2通道
			data_dir=datasets_dir
			channels=channels_selected
			p_names = [i for i in os.listdir(data_dir) if i!="7E3A1B9A-E84C-4B34-A1E4-272FEAED0FA8" and i!="9F6FCBA2-978E-4F6F-937D-6072AECBB5A4"]
			p_names.sort()
			for (idx,name) in enumerate(p_names):
				if os.path.exists(os.path.join(data_dir,name,'X.edf')):
					raw_data = mne.io.read_raw_edf(os.path.join(data_dir,name,'X.edf'))
				elif os.path.exists(os.path.join(data_dir,name,'0.edf')):
					raw_data = mne.io.read_raw_edf(os.path.join(data_dir,name,'0.edf'))
				else:
					raise ValueError(f"File not found for subject {name}/0.edf or {name}/X.edf")
				print(os.path.join(data_dir,name))
				old_freq = raw_data.info['sfreq']
				logger.info(f"raw_data frequency: {old_freq}")

				if args.ICA:
					ICA_channels = ['EEG Fp1-REF', 'EEG Fp2-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG T3-REF', 'EEG T4-REF', 'EOG1', 'EOG2']
					raw_data.set_channel_types({
						'EEG Fp1-REF': 'eeg',
						'EEG Fp2-REF': 'eeg',
						'EEG C3-REF': 'eeg',
						'EEG C4-REF': 'eeg',
						'EEG O1-REF': 'eeg',
						'EEG O2-REF': 'eeg',
						'EEG T3-REF': 'eeg',
						'EEG T4-REF': 'eeg',
						'EOG1': 'eog',
						'EOG2': 'eog',
					})
					raw_data.pick_channels(ICA_channels)
					# raw_data.pick_channels(channels)
					# # 滤波、去伪迹
					# raw_before_ica = raw_data.copy()
					raw_data.load_data()
					raw_data.filter(l_freq=0.5, h_freq=49.9)  # 可选滤波
					ica = mne.preprocessing.ICA(n_components=len(ICA_channels)-2, random_state=97, max_iter='auto')
					ica.fit(raw_data)
					eog_inds, scores = ica.find_bads_eog(raw_data, ch_name=['EOG1', 'EOG2'], threshold=2.5)
					ica.exclude = eog_inds
					raw_data = ica.apply(raw_data)
				
				# Step 6: 可视化前后对比（取前30秒）
				# fig_before = raw_before_ica.plot(start=0, duration=30, title='Before ICA')
				# fig_after = raw_data.plot(start=0, duration=30, title='After ICA')
				# fig_before.set_size_inches(40, 6)  # 宽18英寸，高6英寸（可调）
				# fig_after.set_size_inches(40, 6)
				# fig_before.savefig(os.path.join("result", f'{idx}_before_ica.png'))
				# fig_after.savefig(os.path.join("result", f'{idx}_after_ica.png'))
				raw_data.pick_channels(channels)

				# 保存每个sub的开始时间和结束时间
				self.start_times.append(raw_data.info['meas_date'])
				duration = raw_data.times[-1]  # 最后一个采样点对应的时间（秒）
				end_time = self.start_times[-1] + timedelta(seconds=duration)
				self.end_times.append(end_time)
				# 如果name=2F3A19B9-54EA-4D48-ABDE-59C8EF1DEFD8,去掉最后2分钟
				if name == '2F3A19B9-54EA-4D48-ABDE-59C8EF1DEFD8':
					raw_data.crop(tmin=0, tmax=raw_data.times[-1]-120)
				raw_dataframe = raw_data.to_data_frame()
				xall = np.array(raw_dataframe)[:,1:] # (n,4)

				# --------------------- 插入：逐通道去除异常值 ---------------------
				# 使用3倍标准差剔除异常值（可根据需要修改为其他方法）
				# for ch in range(xall.shape[1]):
				# 	channel_data = xall[:, ch]
				# 	mean = np.mean(channel_data)
				# 	std = np.std(channel_data)
				# 	lower_bound = mean - 3 * std
				# 	upper_bound = mean + 3 * std
				# 	outlier_mask = (channel_data < lower_bound) | (channel_data > upper_bound)
				# 	if np.any(outlier_mask):
				# 		# 可选择使用中值填充异常值，也可以使用均值或前后值插值
				# 		median_val = np.median(channel_data[~outlier_mask])
				# 		channel_data[outlier_mask] = median_val
				# 		logger.info(f"Sub {idx} channel {ch}: {np.sum(outlier_mask)} outliers replaced with median {median_val:.2f}")
				# 	xall[:, ch] = channel_data
				# -------------------------------------------------------------------

				# --------------------- 插入：逐通道去除异常值 ---------------------
				# 去除每个通道中超出 [μ - 3σ, μ + 3σ] 区间的异常值（用邻近值替换）
				# for ch in range(xall.shape[1]):
				# 	channel_data = xall[:, ch]
				# 	mean = np.mean(channel_data)
				# 	std = np.std(channel_data)
				# 	lower_bound = mean - 3 * std
				# 	upper_bound = mean + 3 * std
				# 	outlier_mask = (channel_data < lower_bound) | (channel_data > upper_bound)
					
				# 	if np.any(outlier_mask):
				# 		indices = np.arange(len(channel_data))
				# 		valid_mask = ~outlier_mask
				# 		valid_indices = indices[valid_mask]
				# 		valid_values = channel_data[valid_mask]

				# 		# 最近邻插值：将异常值替换为距离最近的非异常值
				# 		from scipy.interpolate import interp1d
				# 		interpolator = interp1d(valid_indices, valid_values, kind='nearest', fill_value="extrapolate")
				# 		channel_data[outlier_mask] = interpolator(indices[outlier_mask])

				# 		logger.info(f"Sub {idx} channel {ch}: {np.sum(outlier_mask)} outliers replaced with nearest valid value")
					
				# 	xall[:, ch] = channel_data
				# -------------------------------------------------------------------

				# ------------------- 插入：xall 异常值处理（基于 IQR + 邻近值替换） -------------------
				# from scipy.interpolate import interp1d

				# for ch in range(xall.shape[1]):
				# 	channel_data = xall[:, ch]

				# 	# 计算四分位数和 IQR
				# 	Q1 = np.percentile(channel_data, 25)
				# 	Q3 = np.percentile(channel_data, 75)
				# 	IQR = Q3 - Q1

				# 	lower_bound = Q1 - 1.5 * IQR
				# 	upper_bound = Q3 + 1.5 * IQR

				# 	outlier_mask = (channel_data < lower_bound) | (channel_data > upper_bound)

				# 	if np.any(outlier_mask):
				# 		indices = np.arange(len(channel_data))
				# 		valid_mask = ~outlier_mask
				# 		valid_indices = indices[valid_mask]
				# 		valid_values = channel_data[valid_mask]

				# 		# 插值替换异常值
				# 		interpolator = interp1d(valid_indices, valid_values, kind='nearest', fill_value="extrapolate")
				# 		channel_data[outlier_mask] = interpolator(indices[outlier_mask])

				# 		logger.info(f"xall channel {ch}: {np.sum(outlier_mask)} outliers replaced using IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}]")

				# 	xall[:, ch] = channel_data
				# ----------------------------------------------------------------------------------------


				# # 对W、C规范化
				# epochs = xall.shape[0] // (sampling_rate * 30)
				# xall = xall[:epochs*sampling_rate * 30].reshape(epochs, sampling_rate * 30,-1)
				# xall = (xall - np.mean(xall, axis=1, keepdims=True)) / np.std(xall, axis=1, keepdims=True)
				# logger.info(f"**********{np.mean(xall, axis=1, keepdims=True).shape} ")
				# # 复原
				# xall = xall.reshape(-1,xall.shape[-1])

				# 打印极小值、极大值、均值、标准差
				logger.info(f"before norm ：sub {idx} min: {np.min(xall, axis=0)}, max: {np.max(xall, axis=0)}, mean: {np.mean(xall, axis=0)}, std: {np.std(xall, axis=0)}")
				# 滤波
				# from mne.filter import filter_data
				# xall = filter_data(xall.T, sfreq=raw_data.info['sfreq'], l_freq=0.5, h_freq=45).T
				# 对时间维度规范化
				xall = (xall - np.mean(xall, axis=0, keepdims=True)) / np.std(xall, axis=0, keepdims=True)# (n,4)
				logger.info(f"after norm ：sub {idx} min: {np.min(xall, axis=0)}, max: {np.max(xall, axis=0)}, mean: {np.mean(xall, axis=0)}, std: {np.std(xall, axis=0)}")
				# xall = (xall - np.min(xall, axis=0, keepdims=True)) / (np.max(xall, axis=0, keepdims=True) - np.min(xall, axis=0, keepdims=True))# (n,4)
				# 只对第一通道（EEG）标准化
				# xall[:,0] = (xall[:,0] - np.mean(xall[:,0], keepdims=True)) / np.std(xall[:,0], keepdims=True)
				# 只对第二通道（EOG）标准化
				# xall[:,1] = (xall[:,1] - np.mean(xall[:,1], keepdims=True)) / np.std(xall[:,1], keepdims=True)

				if args.real_time:
					# 在实时模式下，在数据头部添加119个epoch
					xall = np.concatenate((np.zeros((119*sampling_rate*30,len(channels))), xall), axis=0)
					# 划窗，大小为input_epoch_num * sampling_rate * 30，步长为sampling_rate * 30
					# 每个窗口的长度（时间点数）
					window_size = self.input_epoch_num * sampling_rate * 30
					step = sampling_rate * 30  # 每次滑动一个 epoch（30秒）
					x_group = []  # 存放窗口数据
					# 滑窗提取窗口
					for start in range(0, xall.shape[0] - window_size + 1, step):
						window = xall[start: start + window_size]  # shape: (T, C)
						window = window.T  # shape: (C, T)
						x_group.append(window)
					# 转为numpy数组，再转为tensor
					x_group = np.stack(x_group, axis=0)  # shape: (W, C, T)
					x_group = torch.from_numpy(x_group).float()
					self.window_num_by_sub.append(x_group.shape[0])  # 记录每个sub的窗口数
					self.last_epoch_num_by_sub.append(0)  # 实时模式下没有最后多余的epoch

				else:
					num_windows = xall.shape[0] // (self.input_epoch_num * sampling_rate * 30)
					self.window_num_by_sub.append(num_windows) # 记录每个sub的120的窗口数
					self.last_epoch_num_by_sub.append((xall.shape[0] % (self.input_epoch_num * sampling_rate * 30) // (sampling_rate * 30))) # 记录最后多余的epoch数
					x_last = None
					if self.last_epoch_num_by_sub[-1] > 0:
						self.window_num_by_sub[-1] += 1
						x_last = xall[-(self.input_epoch_num * sampling_rate * 30):] # (T, 4)
						x_last = np.expand_dims(x_last, axis=0).transpose(0,2,1) # (1,4,T)
					x_group = xall[:num_windows * self.input_epoch_num * sampling_rate * 30].reshape(num_windows,self.input_epoch_num * sampling_rate * 30,-1).transpose(0,2,1) # (num_windows,4,3000*input_epoch_num)
					x_group = np.concatenate((x_group,x_last),axis=0) if x_last is not None else x_group # (num_windows+1,4,3000*input_epoch_num)
					x_group = torch.from_numpy(x_group).float() # （W，C，T）

				taiyang_data.append(x_group)

				if label_available:
					with open(os.path.join(data_dir,name,'label.json'),'r') as f:
						y_group=json.load(f)
					labels = ['W','N1','N2','N3','REM']
					y_group=[labels.index(i['label_name']) for i in y_group]
					self.sleep_times.append(len(y_group))

					if args.real_time:
						# 在实时模式下也进行滑窗（步长为1个epoch）
						# 在数据前面添加了119个 epoch 的 x，因此这里也补119个 -1 标签
						y_group = [0] * 119 + y_group  # 表示无标签的占位符
						# 将标签按滑窗方式切片
						y_windows = []
						for start in range(0, len(y_group) - self.input_epoch_num + 1):
							y_window = y_group[start: start + self.input_epoch_num]  # 长度为 input_epoch_num
							y_windows.append(y_window)

						y_group = torch.tensor(y_windows).float()  # shape: (窗口数, input_epoch_num)
					else:
						y_group = np.array(y_group)	
						if self.last_epoch_num_by_sub[-1] > 0:
							y_last = np.array(y_group[-(self.input_epoch_num):]).reshape((1, self.input_epoch_num))
						y_group = y_group[:num_windows * self.input_epoch_num].reshape((num_windows, self.input_epoch_num))
						y_group = np.concatenate((y_group,y_last),axis=0) if x_last is not None else y_group # (num_windows+1, input_epoch_num)
						
						y_group = torch.from_numpy(y_group).float()
					taiyang_label.append(y_group)	
			# 250630 zmj
			self.x = torch.cat(taiyang_data, dim=0) # （*, num_channels, 3000*input_epoch_num）
			# self.x = torch.concat(taiyang_data, dim=0) # （*, num_channels, 3000*input_epoch_num）
			if label_available:
				self.y = torch.concat(taiyang_label, dim=0)# （*, input_epoch_num）
			
		print("load data finish")

	def gen_testdata(self):
		print(f"===============gen test data only===============")
		test_x,test_y = None,None
		test_x = self.x
		if self.label_available:
			test_y = self.y
			# 打印每个sub的标签数，开始时间，持续时间，结束时间
			for i in range(len(self.sleep_times)):
				logger.info(f"sub {i} num of labels: {self.sleep_times[i]}, start time: {self.start_times[i]}, end time: {self.end_times[i]}, duration: {self.sleep_times[i]/2} min = {self.sleep_times[i]/2/60} h")
		else:
			test_y = np.zeros((test_x.shape[0]))

		test_dataset = MyDataset(test_x, test_y)
		print("test data shape: ", len(test_x))

		return test_dataset
