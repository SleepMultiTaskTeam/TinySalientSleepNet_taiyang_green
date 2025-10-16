import os
import random

# import neurokit2 as nk
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
import yasa
from my_dataset import *
import xml.etree.ElementTree as ET
# from preprocess.loader import load_shhs_xml_stage_label, load_shhs_edf
import pandas as pd

# 250707 by MJZ
def smooth_labels(y, num_classes=5, smoothing=0.1):
    """
    支持二维输入的标签平滑。
    y: (batch_size, time_steps) 的整数标签
    返回: (batch_size, time_steps, num_classes) 的平滑 soft label
    """
    assert 0 <= smoothing < 1
    y = np.array(y).astype(int)  # shape: (B, T)

    batch_size, time_steps = y.shape
    y_smooth = np.full((batch_size, time_steps, num_classes), smoothing / num_classes, dtype=np.float32)

    for i in range(batch_size):
        for j in range(time_steps):
            y_smooth[i, j, y[i, j]] += (1.0 - smoothing)
    
    return y_smooth  # shape: (B, T, C)



class DataPreprocesserSHHS:
	def __init__(self, par_dict, datasets_dir, channels_selected, dataset, test_dataset, input_epoch_num, sampling_rate, fold_num,test_size,val_size, yasa_feature, channels_selected2, data_double):
		self.fold_num = fold_num
		self.test_size = test_size
		self.val_size = val_size
		self.test_dataset = test_dataset
		self.input_epoch_num = input_epoch_num
		self.yasa_feature = yasa_feature
		self.epoch_num_by_sub = []

		data = {}
		data['train'], data['val'], data['test']=[],[],[]
		data_yasa = {}
		data_yasa['train'], data_yasa['val'], data_yasa['test']=[],[],[]
		label = {}
		label['train'], label['val'], label['test']=[],[],[]

		taiyang_data=[]
		taiyang_label=[]
		for d in dataset:
			if d=='ISRUC1':
				data_dir=datasets_dir[d]
				channels=channels_selected[d]
				# 读取data_dir中所有以.mat结尾的文件
				p_names = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
				# # 打乱顺序
				# random.shuffle(p_names)
				p_names.sort()
				# 按8：1：1划分数据集
				val_num=round(len(p_names)*self.val_size)
				test_num=round(len(p_names)*self.test_size)
				train_num=len(p_names)-val_num-test_num
				train_p_names=p_names[:train_num]
				val_p_names=p_names[train_num:train_num+val_num]
				test_p_names=p_names[train_num+val_num:]
				print("train_num: ",train_num,"val_num: ",val_num,"test_num: ",test_num)
				tvt={'train':train_p_names,'val':val_p_names,'test':test_p_names}
				for c,names in tvt.items():
					for name in names:
						raw_data = sio.loadmat(os.path.join(data_dir, name))
						print(os.path.join(data_dir, name))
						# 将所选通道的数据拼接
						xall = np.concatenate([raw_data[i][:,np.newaxis,:] for i in channels],axis=1) # (n,4,6000)
						# 降采样；200Hz->100Hz
						xall = signal.resample(xall, int(xall.shape[2]//2), axis=2) # (n,4,3000)
						# 提取yasa特征
						features = None
						if self.yasa_feature!="False":
							# 需要(channel,timestamp)格式的x
							x = xall.transpose(1,0,2).reshape(len(channels),-1)
							info = mne.create_info(ch_names=channels, sfreq=sampling_rate, ch_types=['eeg', 'eog'])
							raw = mne.io.RawArray(x, info)
							sls = yasa.SleepStaging(raw, eeg_name=channels[0], eog_name=channels[1])
							features = sls.get_features()

						# 对每个窗口的每个通道进行规范化
						xall = (xall - np.mean(xall, axis=(0,2), keepdims=True)) / np.std(xall, axis=(0,2), keepdims=True)# (n,4,3000)
						index=name.split('.')[0][7:] # 获取文件名中的index,用于获取标签
						label_name=index+'_1.npy'
						stage_label=np.load(os.path.join(data_dir,'label',label_name)) # (n,)
						x_group,y_group=self.get_data_by_input_epoch_num(xall,stage_label,input_epoch_num)

						if c=="test":
							if self.test_dataset == 'self' or self.test_dataset == 'ISRUC':
								data[c].append(x_group)
								label[c].append(y_group)
								if self.yasa_feature!="False":
									data_yasa[c].append(features)
							else:
								data['train'].append(x_group)
								label['train'].append(y_group)
								if self.yasa_feature!="False":
									data_yasa['train'].append(features)
						else:
							data[c].append(x_group)
							label[c].append(y_group)
							if self.yasa_feature!="False":
								data_yasa[c].append(features)
				if data_double:
					data_dir=datasets_dir[d]
					channels=channels_selected2[d]
					# 读取data_dir中所有以.mat结尾的文件
					p_names = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
					# # 打乱顺序
					# random.shuffle(p_names)
					p_names.sort()
					# 按8：1：1划分数据集
					val_num=round(len(p_names)*self.val_size)
					test_num=round(len(p_names)*self.test_size)
					train_num=len(p_names)-val_num-test_num
					train_p_names=p_names[:train_num]
					val_p_names=p_names[train_num:train_num+val_num]
					test_p_names=p_names[train_num+val_num:]
					print("train_num: ",train_num,"val_num: ",val_num,"test_num: ",test_num)
					tvt={'train':train_p_names,'val':val_p_names,'test':test_p_names}
					for c,names in tvt.items():
						for name in names:
							raw_data = sio.loadmat(os.path.join(data_dir, name))
							print(os.path.join(data_dir, name))
							# 将所选通道的数据拼接
							xall = np.concatenate([raw_data[i][:,np.newaxis,:] for i in channels],axis=1) # (n,4,6000)
							# 降采样；200Hz->100Hz
							xall = signal.resample(xall, int(xall.shape[2]//2), axis=2) # (n,4,3000)
							# 提取yasa特征
							features = None
							if self.yasa_feature!="False":
								# 需要(channel,timestamp)格式的x
								x = xall.transpose(1,0,2).reshape(len(channels),-1)
								info = mne.create_info(ch_names=channels, sfreq=sampling_rate, ch_types=['eeg', 'eog'])
								raw = mne.io.RawArray(x, info)
								sls = yasa.SleepStaging(raw, eeg_name=channels[0], eog_name=channels[1])
								features = sls.get_features()

							# 对每个窗口的每个通道进行规范化
							xall = (xall - np.mean(xall, axis=(0,2), keepdims=True)) / np.std(xall, axis=(0,2), keepdims=True)# (n,4,3000)
							index=name.split('.')[0][7:] # 获取文件名中的index,用于获取标签
							label_name=index+'_1.npy'
							stage_label=np.load(os.path.join(data_dir,'label',label_name)) # (n,)
							x_group,y_group=self.get_data_by_input_epoch_num(xall,stage_label,input_epoch_num)

							if c=="test":
								if self.test_dataset == 'self' or self.test_dataset == 'ISRUC':
									data[c].append(x_group)
									label[c].append(y_group)
									if self.yasa_feature!="False":
										data_yasa[c].append(features)
								else:
									data['train'].append(x_group)
									label['train'].append(y_group)
									if self.yasa_feature!="False":
										data_yasa['train'].append(features)
							else:
								data[c].append(x_group)
								label[c].append(y_group)
								if self.yasa_feature!="False":
									data_yasa[c].append(features)
							
			elif d=="ISRUC3":
				data_dir=datasets_dir[d]
				channels=channels_selected[d]
				# 读取data_dir中所有以.mat结尾的文件
				p_names = [f for f in os.listdir(data_dir) if f.endswith('.mat') ]
				# # 打乱顺序
				# random.shuffle(p_names)
				p_names.sort()
				# 按8：1：1划分数据集
				val_num=round(len(p_names)*self.val_size)
				test_num=round(len(p_names)*self.test_size)
				train_num=len(p_names)-val_num-test_num
				train_p_names=p_names[:train_num]
				val_p_names=p_names[train_num:train_num+val_num]
				test_p_names=p_names[train_num+val_num:]
				print("train_num: ",train_num,"val_num: ",val_num,"test_num: ",test_num)
				tvt={'train':train_p_names,'val':val_p_names,'test':test_p_names}
				for c,names in tvt.items():
					for name in names:
						raw_data = sio.loadmat(os.path.join(data_dir, name))
						print(os.path.join(data_dir, name))
						# 将所选通道的数据拼接
						xall = np.concatenate([raw_data[i][:,np.newaxis,:] for i in channels],axis=1) # (n,4,6000)
						# 降采样；200Hz->100Hz
						xall = signal.resample(xall, int(xall.shape[2]//2), axis=2) # (n,4,3000)
						# 提取yasa特征
						features = None
						if self.yasa_feature!="False":
							# 需要(channel,timestamp)格式的x
							x = xall.transpose(1,0,2).reshape(len(channels),-1)
							info = mne.create_info(ch_names=channels, sfreq=sampling_rate, ch_types=['eeg', 'eog'])
							raw = mne.io.RawArray(x, info, verbose=False)
							sls = yasa.SleepStaging(raw, eeg_name=channels[0], eog_name=channels[1])
							features = sls.get_features()
							features = np.array(features)# (n,116)
							print(features.shape)
							num_epochs = features.shape[0] // self.input_epoch_num
							features = features[:num_epochs * self.input_epoch_num].reshape((num_epochs, self.input_epoch_num, -1)).transpose(0, 2, 1) # (n',116,input_epoch_num)
							features = torch.from_numpy(features).float()
							# 规范化
							features = (features - torch.mean(features, dim=(0,2), keepdim=True)) / torch.std(features, dim=(0,2), keepdim=True)
						# 对每个窗口的每个通道进行规范化
						xall = (xall - np.mean(xall, axis=(0,2), keepdims=True)) / np.std(xall, axis=(0,2), keepdims=True)# (n,4,3000)
						index=name.split('.')[0][7:] # 获取文件名中的index,用于获取标签
						label_name=index+'-Label.mat'
						stage_label=sio.loadmat(os.path.join(data_dir,'label',label_name))['label'].reshape(-1) # (n,)
						x_group,y_group=self.get_data_by_input_epoch_num(xall,stage_label,input_epoch_num)

						if c=="test":
							if self.test_dataset == 'self' or self.test_dataset == 'ISRUC':
								data[c].append(x_group)
								label[c].append(y_group)
								if self.yasa_feature!="False":
									data_yasa[c].append(features)
							else:
								data['train'].append(x_group)
								label['train'].append(y_group)
								if self.yasa_feature!="False":
									data_yasa['train'].append(features)
						else:
							data[c].append(x_group)
							label[c].append(y_group)
							if self.yasa_feature!="False":
								data_yasa[c].append(features)
				if data_double:
					data_dir=datasets_dir[d]
					channels=channels_selected2[d]
					# 读取data_dir中所有以.mat结尾的文件
					p_names = [f for f in os.listdir(data_dir) if f.endswith('.mat') ]
					# # 打乱顺序
					# random.shuffle(p_names)
					p_names.sort()
					# 按8：1：1划分数据集
					val_num=round(len(p_names)*self.val_size)
					test_num=round(len(p_names)*self.test_size)
					train_num=len(p_names)-val_num-test_num
					train_p_names=p_names[:train_num]
					val_p_names=p_names[train_num:train_num+val_num]
					test_p_names=p_names[train_num+val_num:]
					print("train_num: ",train_num,"val_num: ",val_num,"test_num: ",test_num)
					tvt={'train':train_p_names,'val':val_p_names,'test':test_p_names}
					for c,names in tvt.items():
						for name in names:
							raw_data = sio.loadmat(os.path.join(data_dir, name))
							print(os.path.join(data_dir, name))
							# 将所选通道的数据拼接
							xall = np.concatenate([raw_data[i][:,np.newaxis,:] for i in channels],axis=1) # (n,4,6000)
							# 降采样；200Hz->100Hz
							xall = signal.resample(xall, int(xall.shape[2]//2), axis=2) # (n,4,3000)
							# 提取yasa特征
							features = None
							if self.yasa_feature!="False":
								# 需要(channel,timestamp)格式的x
								x = xall.transpose(1,0,2).reshape(len(channels),-1)
								info = mne.create_info(ch_names=channels, sfreq=sampling_rate, ch_types=['eeg', 'eog'])
								raw = mne.io.RawArray(x, info, verbose=False)
								sls = yasa.SleepStaging(raw, eeg_name=channels[0], eog_name=channels[1])
								features = sls.get_features()
								features = np.array(features)# (n,116)
								print(features.shape)
								num_epochs = features.shape[0] // self.input_epoch_num
								features = features[:num_epochs * self.input_epoch_num].reshape((num_epochs, self.input_epoch_num, -1)).transpose(0, 2, 1) # (n',116,input_epoch_num)
								features = torch.from_numpy(features).float()
								# 规范化
								features = (features - torch.mean(features, dim=(0,2), keepdim=True)) / torch.std(features, dim=(0,2), keepdim=True)
							# 对每个窗口的每个通道进行规范化
							xall = (xall - np.mean(xall, axis=(0,2), keepdims=True)) / np.std(xall, axis=(0,2), keepdims=True)# (n,4,3000)
							index=name.split('.')[0][7:] # 获取文件名中的index,用于获取标签
							label_name=index+'-Label.mat'
							stage_label=sio.loadmat(os.path.join(data_dir,'label',label_name))['label'].reshape(-1) # (n,)
							x_group,y_group=self.get_data_by_input_epoch_num(xall,stage_label,input_epoch_num)

							if c=="test":
								if self.test_dataset == 'self' or self.test_dataset == 'ISRUC':
									data[c].append(x_group)
									label[c].append(y_group)
									if self.yasa_feature!="False":
										data_yasa[c].append(features)
								else:
									data['train'].append(x_group)
									label['train'].append(y_group)
									if self.yasa_feature!="False":
										data_yasa['train'].append(features)
							else:
								data[c].append(x_group)
								label[c].append(y_group)
								if self.yasa_feature!="False":
									data_yasa[c].append(features)		
			elif d=="SHHS":
				data_dir=datasets_dir[d]
				channels=channels_selected[d]
				channel=['EEG', "EEG(sec)", 'EOG(L)', 'EMG']#["C4","C3","EOGL","EMG"]
				channel_index=[channel.index(c) for c in channels]
				p_names = os.listdir(data_dir)
				p_names.sort()
				start = 1
				end = 501
				print(start,end)
				p_names = p_names[start:end]
				# random.shuffle(p_names)
				p_names.sort()
				# 按8：1：1划分数据集
				val_num=round(len(p_names)*self.val_size)
				test_num=round(len(p_names)*self.test_size)
				train_num=len(p_names)-val_num-test_num
				train_p_names=p_names[:train_num]
				val_p_names=p_names[train_num:train_num+val_num]
				test_p_names=p_names[train_num+val_num:]
				print("train_num: ",train_num,"val_num: ",val_num,"test_num: ",test_num)
				tvt={'train':train_p_names,'val':val_p_names,'test':test_p_names}
				for c,names in tvt.items():
					for name in names:
						with open(os.path.join(data_dir, name), 'rb') as f:
							raw_data = pkl.load(f)
						print(os.path.join(data_dir, name))
						# 将所选通道的数据拼接
						xall = raw_data['new_xall'][:,channel_index] # (n,4)
						# 提取yasa特征
						features = None
						if self.yasa_feature!="False":
							# 需要(channel,timestamp)格式的x
							x = xall.transpose(1,0)
							info = mne.create_info(ch_names=channels, sfreq=sampling_rate, ch_types=['eeg', 'eog'])
							raw = mne.io.RawArray(x, info)
							sls = yasa.SleepStaging(raw, eeg_name=channels[0], eog_name=channels[1])
							features = sls.get_features()
						# 对时间维度规范化
						xall = (xall - np.mean(xall, axis=0, keepdims=True)) / np.std(xall, axis=0, keepdims=True)# (n,4)
						num_epochs = xall.shape[0] // (self.input_epoch_num * sampling_rate * 30)
						x_group = xall[:num_epochs * self.input_epoch_num * sampling_rate * 30].reshape(num_epochs,self.input_epoch_num * sampling_rate * 30,-1).transpose(0,2,1)
						
						y_group = raw_data["stage_label"]
						y_group = y_group[:num_epochs * self.input_epoch_num].reshape((num_epochs, self.input_epoch_num))
						x_group, y_group = torch.from_numpy(x_group).float(), torch.from_numpy(y_group).float()

						if c=="test":
							if self.test_dataset == 'self' or self.test_dataset == 'SHHS':
								data[c].append(x_group)
								label[c].append(y_group)
								self.epoch_num_by_sub.append(y_group.shape[0]* input_epoch_num) # 记录每个sub的标签数
								if self.yasa_feature!="False":
									data_yasa[c].append(features)
							else:
								data['train'].append(x_group)
								label['train'].append(y_group)
								if self.yasa_feature!="False":
									data_yasa['train'].append(features)
						else:
							data[c].append(x_group)
							label[c].append(y_group)
							if self.yasa_feature!="False":
								data_yasa[c].append(features)
			elif d=="MASS":
				data_dir=datasets_dir[d]
				channels=channels_selected[d]
				channel=['FP1','FP2','Fz' , 'F3' , 'F4' , 'F7' , 'F8' , 'C3' , 'C4' , 'T3' , 'T4' , 'Pz' , 'P3' , 'P4' , 'T5' , 'T6' , 'Oz' , 'O1' , 'O2' , 'EogL' , 'EogR' , 'Emg1' , 'Emg2' , 'Emg3' , 'Ecg' ]
				channel_index=[channel.index(c) for c in channels]
				# 读取data_dir中所有以-Datasub.mat结尾的文件
				p_names = [f for f in os.listdir(data_dir) if f.endswith('-Datasub.mat')]
				# # 打乱顺序
				# random.shuffle(p_names)
				p_names.sort()
				# 按8：1：1划分数据集
				val_num=round(len(p_names)*self.val_size)
				test_num=round(len(p_names)*self.test_size)
				train_num=len(p_names)-val_num-test_num
				train_p_names=p_names[:train_num]
				val_p_names=p_names[train_num:train_num+val_num]
				test_p_names=p_names[train_num+val_num:]
				print("train_num: ",train_num,"val_num: ",val_num,"test_num: ",test_num)
				tvt={'train':train_p_names,'val':val_p_names,'test':test_p_names}
				for c,names in tvt.items():
					for name in names:
						raw_data = sio.loadmat(os.path.join(data_dir, name))
						print(os.path.join(data_dir, name))
						# 将所选通道的数据拼接
						xall = raw_data['PSG'][:,channel_index,:] # (n,4,3000)
						# 提取yasa特征
						features = None
						if self.yasa_feature!="False":
							# 需要(channel,timestamp)格式的x
							x = xall.transpose(1,0,2).reshape(len(channels),-1)
							info = mne.create_info(ch_names=channels, sfreq=sampling_rate, ch_types=['eeg', 'eog'])
							raw = mne.io.RawArray(x, info)
							sls = yasa.SleepStaging(raw, eeg_name=channels[0], eog_name=channels[1])
							features = sls.get_features()
						# 对时间维度规范化
						xall = (xall - np.mean(xall, axis=(0,2), keepdims=True)) / np.std(xall, axis=(0,2), keepdims=True)# (n,4,3000)
						
						index=name[:10] # 获取文件名中的index,用于获取标签
						label_name=index+'-Label.mat'
						stage_label=sio.loadmat(os.path.join(data_dir,label_name))['label'] # (n,5)
						stage_label = np.argmax(stage_label, axis=1) # (n,)
						x_group,y_group=self.get_data_by_input_epoch_num(xall,stage_label,input_epoch_num)

						if c=="test":
							if self.test_dataset == 'self' or self.test_dataset == 'MASS':
								data[c].append(x_group)
								label[c].append(y_group)
								self.epoch_num_by_sub.append(y_group.shape[0]* input_epoch_num) # 记录每个sub的标签数
								if self.yasa_feature!="False":
									data_yasa[c].append(features)
							else:
								data['train'].append(x_group)
								label['train'].append(y_group)
								if self.yasa_feature!="False":
									data_yasa['train'].append(features)
						else:
							data[c].append(x_group)
							label[c].append(y_group)
							if self.yasa_feature!="False":
								data_yasa[c].append(features)
				if data_double:
					data_dir=datasets_dir[d]
					channels=channels_selected2[d]
					channel=['FP1','FP2','Fz' , 'F3' , 'F4' , 'F7' , 'F8' , 'C3' , 'C4' , 'T3' , 'T4' , 'Pz' , 'P3' , 'P4' , 'T5' , 'T6' , 'Oz' , 'O1' , 'O2' , 'EogL' , 'EogR' , 'Emg1' , 'Emg2' , 'Emg3' , 'Ecg' ]
					channel_index=[channel.index(c) for c in channels]
					# 读取data_dir中所有以-Datasub.mat结尾的文件
					p_names = [f for f in os.listdir(data_dir) if f.endswith('-Datasub.mat')]
					# # 打乱顺序
					# random.shuffle(p_names)
					p_names.sort()
					# 按8：1：1划分数据集
					val_num=round(len(p_names)*self.val_size)
					test_num=round(len(p_names)*self.test_size)
					train_num=len(p_names)-val_num-test_num
					train_p_names=p_names[:train_num]
					val_p_names=p_names[train_num:train_num+val_num]
					test_p_names=p_names[train_num+val_num:]
					print("train_num: ",train_num,"val_num: ",val_num,"test_num: ",test_num)
					tvt={'train':train_p_names,'val':val_p_names,'test':test_p_names}
					for c,names in tvt.items():
						for name in names:
							raw_data = sio.loadmat(os.path.join(data_dir, name))
							print(os.path.join(data_dir, name))
							# 将所选通道的数据拼接
							xall = raw_data['PSG'][:,channel_index,:] # (n,4,3000)
							# 提取yasa特征
							features = None
							if self.yasa_feature!="False":
								# 需要(channel,timestamp)格式的x
								x = xall.transpose(1,0,2).reshape(len(channels),-1)
								info = mne.create_info(ch_names=channels, sfreq=sampling_rate, ch_types=['eeg', 'eog'])
								raw = mne.io.RawArray(x, info)
								sls = yasa.SleepStaging(raw, eeg_name=channels[0], eog_name=channels[1])
								features = sls.get_features()
							# 对时间维度规范化
							xall = (xall - np.mean(xall, axis=(0,2), keepdims=True)) / np.std(xall, axis=(0,2), keepdims=True)# (n,4,3000)
							
							index=name[:10] # 获取文件名中的index,用于获取标签
							label_name=index+'-Label.mat'
							stage_label=sio.loadmat(os.path.join(data_dir,label_name))['label'] # (n,5)
							stage_label = np.argmax(stage_label, axis=1) # (n,)
							x_group,y_group=self.get_data_by_input_epoch_num(xall,stage_label,input_epoch_num)

							if c=="test":
								if self.test_dataset == 'self' or self.test_dataset == 'MASS':
									data[c].append(x_group)
									label[c].append(y_group)
									self.epoch_num_by_sub.append(y_group.shape[0]* input_epoch_num) # 记录每个sub的标签数
									if self.yasa_feature!="False":
										data_yasa[c].append(features)
								else:
									data['train'].append(x_group)
									label['train'].append(y_group)
									if self.yasa_feature!="False":
										data_yasa['train'].append(features)
							else:
								data[c].append(x_group)
								label[c].append(y_group)
								if self.yasa_feature!="False":
									data_yasa[c].append(features)
			elif d == "SLEEPEDF153":
				data_dir = datasets_dir[d]
				channels = channels_selected[d]
				channel = ['Fpz-Cz','Pz-Oz','EOG']
				channel_index = [channel.index(c) for c in channels]
				
				p_names = [f for f in os.listdir(data_dir)]
				# random.shuffle(p_names)
				p_names.sort()
				val_num = round(len(p_names) * self.val_size)
				test_num = round(len(p_names) * self.test_size)
				train_num = len(p_names) - val_num - test_num
				train_p_names = p_names[:train_num]
				val_p_names = p_names[train_num:train_num + val_num]
				test_p_names = p_names[train_num + val_num:]
				print("train_num: ", train_num, "val_num: ", val_num, "test_num: ", test_num)
				tvt = {'train': train_p_names, 'val': val_p_names, 'test': test_p_names}

				# # Initialize lists for data and label
				# data = {'train': [], 'val': [], 'test': []}
				# label = {'train': [], 'val': [], 'test': []}

				for c, names in tvt.items():
					for name in names:
						print(os.path.join(data_dir, name))
						if not os.path.exists(os.path.join(data_dir, name)):
							print(f"The file {os.path.join(data_dir, name)} does not exist.")
							continue
						try:
							npz_file = np.load(os.path.join(data_dir, name), allow_pickle=True)
						except IOError as e:
							print(f"Failed to load data from {os.path.join(data_dir, name)}: {e}")
							continue

						x = npz_file['x'][:, :, channel_index].transpose(0, 2, 1) # (n,4,3000)
						# 提取yasa特征
						features = None
						if self.yasa_feature!="False":
							# 需要(channel,timestamp)格式的x
							xall = x.transpose(1,0,2).reshape(len(channels),-1)
							info = mne.create_info(ch_names=channels, sfreq=sampling_rate, ch_types=['eeg', 'eog'])
							raw = mne.io.RawArray(xall, info)
							sls = yasa.SleepStaging(raw, eeg_name=channels[0], eog_name=channels[1])
							features = sls.get_features()
						# 规范化
						x = (x - np.mean(x, axis=(0,2), keepdims=True)) / np.std(x, axis=(0,2), keepdims=True)# (n,4,3000)
						y = npz_file['y']

						# print(x.shape, y.shape)
						x_group,y_group=self.get_data_by_input_epoch_num(x,y,input_epoch_num)
						# print(x_group.shape, y_group.shape)

						if c == "test":
							if self.test_dataset == 'self' or self.test_dataset == 'SLEEPEDF153':
								data[c].append(x_group)
								label[c].append(y_group)
								self.epoch_num_by_sub.append(y_group.shape[0] * input_epoch_num) # 记录每个sub的标签数
								if self.yasa_feature!="False":
									data_yasa[c].append(features)
							else:
								data['train'].append(x_group)
								label['train'].append(y_group)
								if self.yasa_feature!="False":
									data_yasa['train'].append(features)
						else:
							data[c].append(x_group)
							label[c].append(y_group)
							if self.yasa_feature!="False":
								data_yasa[c].append(features)
				# if data_double:
					# continue
					# data_dir = datasets_dir[d]
					# channels = channels_selected2[d]
					# channel = ['Fpz-Cz','EOG','EMG']
					# channel_index = [channel.index(c) for c in channels]
					
					# p_names = [f for f in os.listdir(data_dir)]
					# # random.shuffle(p_names)
					# p_names.sort()
					# val_num = round(len(p_names) * self.val_size)
					# test_num = round(len(p_names) * self.test_size)
					# train_num = len(p_names) - val_num - test_num
					# train_p_names = p_names[:train_num]
					# val_p_names = p_names[train_num:train_num + val_num]
					# test_p_names = p_names[train_num + val_num:]
					# print("train_num: ", train_num, "val_num: ", val_num, "test_num: ", test_num)
					# tvt = {'train': train_p_names, 'val': val_p_names, 'test': test_p_names}

					# # # Initialize lists for data and label
					# # data = {'train': [], 'val': [], 'test': []}
					# # label = {'train': [], 'val': [], 'test': []}

					# for c, names in tvt.items():
					# 	for name in names:
					# 		print(os.path.join(data_dir, name))
					# 		if not os.path.exists(os.path.join(data_dir, name)):
					# 			print(f"The file {os.path.join(data_dir, name)} does not exist.")
					# 			continue
					# 		try:
					# 			npz_file = np.load(os.path.join(data_dir, name), allow_pickle=True)
					# 		except IOError as e:
					# 			print(f"Failed to load data from {os.path.join(data_dir, name)}: {e}")
					# 			continue

					# 		x = npz_file['x'][:, :, channel_index].transpose(0, 2, 1) # (n,4,3000)
					# 		# 提取yasa特征
					# 		features = None
					# 		if self.yasa_feature!="False":
					# 			# 需要(channel,timestamp)格式的x
					# 			xall = x.transpose(1,0,2).reshape(len(channels),-1)
					# 			info = mne.create_info(ch_names=channels, sfreq=sampling_rate, ch_types=['eeg', 'eog'])
					# 			raw = mne.io.RawArray(xall, info)
					# 			sls = yasa.SleepStaging(raw, eeg_name=channels[0], eog_name=channels[1])
					# 			features = sls.get_features()
					# 		# 规范化
					# 		x = (x - np.mean(x, axis=(0,2), keepdims=True)) / np.std(x, axis=(0,2), keepdims=True)# (n,4,3000)
					# 		y = npz_file['y']

					# 		# print(x.shape, y.shape)
					# 		x_group,y_group=self.get_data_by_input_epoch_num(x,y,input_epoch_num)
					# 		# print(x_group.shape, y_group.shape)

					# 		if c == "test":
					# 			if self.test_dataset == 'self' or self.test_dataset == 'SLEEPEDF153':
					# 				data[c].append(x_group)
					# 				label[c].append(y_group)
					# 				self.epoch_num_by_sub.append(y_group.shape[0] * input_epoch_num) # 记录每个sub的标签数
					# 				if self.yasa_feature!="False":
					# 					data_yasa[c].append(features)
					# 			else:
					# 				data['train'].append(x_group)
					# 				label['train'].append(y_group)
					# 				if self.yasa_feature!="False":
					# 					data_yasa['train'].append(features)
					# 		else:
					# 			data[c].append(x_group)
					# 			label[c].append(y_group)
					# 			if self.yasa_feature!="False":
					# 				data_yasa[c].append(features)
			elif d=="taiyang":
				# 读取仅做测试，4通道/2通道
				data_dir=datasets_dir[d]
				channels=channels_selected[d]
				p_names = [i for i in os.listdir(data_dir) if i!="7E3A1B9A-E84C-4B34-A1E4-272FEAED0FA8" and i!="9F6FCBA2-978E-4F6F-937D-6072AECBB5A4"]
				p_names.sort()
				for (idx,name) in enumerate(p_names):

					path = os.path.join(data_dir, name, 'X.edf')

					if not os.path.exists(path):
						path = os.path.join(data_dir, name, '0.edf')

					print("使用的路径是:", path)
					raw_data = mne.io.read_raw_edf(path)
					# print(os.path.join(data_dir,name,'X.edf'))
					raw_data.pick_channels(channels)
					# 如果name=2F3A19B9-54EA-4D48-ABDE-59C8EF1DEFD8,去掉最后2分钟
					if name == '2F3A19B9-54EA-4D48-ABDE-59C8EF1DEFD8':
						raw_data.crop(tmin=0, tmax=raw_data.times[-1]-120)
					raw_dataframe = raw_data.to_data_frame()
					xall = np.array(raw_dataframe)[:,1:] # (n,4)
					print(f'【xall shape:】{xall.shape}')
					# 提取yasa特征
					features = None
					if self.yasa_feature!="False":
						# 需要(channel,timestamp)格式的x
						x = xall.transpose(1,0)
						info = mne.create_info(ch_names=channels, sfreq=sampling_rate, ch_types=['eeg', 'eog'])
						raw = mne.io.RawArray(x, info)
						sls = yasa.SleepStaging(raw, eeg_name=channels[0], eog_name=channels[1])
						features = sls.get_features()
					# 对时间维度规范化
					xall = (xall - np.mean(xall, axis=0, keepdims=True)) / np.std(xall, axis=0, keepdims=True)# (n,4)
					num_epochs = xall.shape[0] // (self.input_epoch_num * sampling_rate * 30)
					self.epoch_num_by_sub.append(num_epochs*self.input_epoch_num) # 记录每个sub的标签数
					x_group = xall[:num_epochs * self.input_epoch_num * sampling_rate * 30].reshape(num_epochs,self.input_epoch_num * sampling_rate * 30,-1).transpose(0,2,1) # (num_epoch,4,3000)

					with open(os.path.join(data_dir,name,'label.json'),'r') as f:
						y_group=json.load(f)
					labels = ['W','N1','N2','N3','REM']
					y_group=[labels.index(i['label_name']) for i in y_group]
					y_group = np.array(y_group)
					# with open('/mnt/nfsData18/ZhangShaoqi/CODES/SSSC_1/try/yasa_2c_ypre_dict.pkl','rb') as f:
					# 	y_group=pkl.load(f)
					# y_group = y_group[idx]

					y_group = y_group[:num_epochs * self.input_epoch_num].reshape((num_epochs, self.input_epoch_num))
					x_group = torch.from_numpy(x_group).float()
					y_group = torch.from_numpy(y_group).float()

					taiyang_data.append(x_group)
					taiyang_label.append(y_group)
					if self.yasa_feature!="False":
						features = torch.from_numpy(features).float()
						data_yasa['test'].append(features)	

				self.x = torch.cat(taiyang_data, dim=0)
				self.y = torch.cat(taiyang_label, dim=0)
				if self.yasa_feature!="False":
					self.x_yasa = torch.cat(data_yasa['test'], dim=0)	
				return	
			
			elif d=="ccshs":
				# 2通道
				data_dir=datasets_dir[d]
				channels=channels_selected[d]

				index_path = "/mnt/nfsData18/ZhangShaoqi/CODES/SSSC_1/nsrrid_training_testing.csv"
				da = "CCSHS"
				index_df = pd.read_csv(index_path)
				train_names = index_df[(index_df['dataset']==da) & (index_df['set']=="training")]['subj'].values
				train_names.sort()
				train_num = len(train_names)
				val_p_names = train_names[-round(train_num * self.val_size):]
				train_p_names = train_names[:len(train_names)-round(train_num * self.val_size)]
				val_num = len(val_p_names)
				train_num = len(train_p_names)
				test_p_names = index_df[(index_df['dataset']==da) & (index_df['set']=="testing")]['subj'].values
				test_num = len(test_p_names)
				# 构造 ccshs-trec-1800001.edf 格式的文件名
				train_p_names = [f"ccshs-trec-{i}.edf" for i in train_p_names]
				val_p_names = [f"ccshs-trec-{i}.edf" for i in val_p_names]
				test_p_names = [f"ccshs-trec-{i}.edf" for i in test_p_names]

				print("train_num: ", train_num, "val_num: ", val_num, "test_num: ", test_num)
				tvt = {'train': train_p_names, 'val': val_p_names, 'test': test_p_names}

				for c, names in tvt.items():
					for (idx,name) in enumerate(names):
						raw_data = mne.io.read_raw_edf(os.path.join(data_dir,'edfs_new',name))
						print(os.path.join(data_dir,'edfs_new',name))
						raw_data.pick_channels(channels)
						xall = raw_data.get_data() # (2,n)
						# 降采样；128Hz->100Hz
						xall = signal.resample(xall, int(xall.shape[1]*100//128), axis=1)
						# 对时间维度规范化
						xall = (xall - np.mean(xall, axis=1, keepdims=True)) / np.std(xall, axis=1, keepdims=True)# (2,n)
						
						# 读取标签
						tree = ET.parse(os.path.join(data_dir,'annotations-events-profusion',name.split('.')[0]+'-profusion.xml'))
						root = tree.getroot()
						# 提取其中<SleepStage>标签
						labels = []
						for child in root:
							if child.tag == 'SleepStages':
								for stage in child:
									labels.append(int(stage.text))
						
						y_group = np.array(labels)
						# 将其中的4改为3
						y_group[y_group==4] = 3
						# 将其中的5改为4
						y_group[y_group==5] = 4
						# 如果存在不属于[0,1,2,3,4]的（x,y）删去，即6或9
						# 获取不属于[0,1,2,3,4]的index
						invalid_index = np.where(np.logical_not(np.isin(y_group, [0,1,2,3,4])))
						# 删除
						y_group = np.delete(y_group, invalid_index[0], axis=0)
						nums = xall.shape[1]//(sampling_rate*30)
						assert  nums == len(labels), f"epoch nums: {nums} != len(labels): {len(labels)}"
						xall = xall.reshape(2,nums,sampling_rate*30)
						xall = np.delete(xall, invalid_index[0], axis=1)

						# 划窗
						num_windows = xall.shape[1] // (self.input_epoch_num)
						# self.epoch_num_by_sub.append(num_windows*self.input_epoch_num) # 记录每个sub的标签数
						x_group = xall[:,:num_windows * self.input_epoch_num ,:].reshape(-1,num_windows,self.input_epoch_num * sampling_rate * 30).transpose(1,0,2) # (num_windows,2,120*3000)
						y_group = y_group[:num_windows * self.input_epoch_num].reshape((num_windows, self.input_epoch_num)) # (num_windows,input_epoch_num)

						x_group = torch.from_numpy(x_group).float()
						y_group = torch.from_numpy(y_group).float()

						if c=="test":
							if self.test_dataset == 'self' or self.test_dataset == 'ccshs':
								data[c].append(x_group)
								label[c].append(y_group)
								if self.yasa_feature!="False":
									data_yasa[c].append(features)
							else:
								data['train'].append(x_group)
								label['train'].append(y_group)
								if self.yasa_feature!="False":
									data_yasa['train'].append(features)
						else:
							data[c].append(x_group)
							label[c].append(y_group)
							if self.yasa_feature!="False":
								data_yasa[c].append(features)
			
			elif d=="cfs":
				# 2通道
				data_dir=datasets_dir[d]
				channels=channels_selected[d]
				
				index_path = "/mnt/nfsData18/ZhangShaoqi/CODES/SSSC_1/nsrrid_training_testing.csv"
				da = "CFS"
				index_df = pd.read_csv(index_path)
				train_names = index_df[(index_df['dataset']==da) & (index_df['set']=="training")]['subj'].values
				train_names.sort()
				train_num = len(train_names)
				val_p_names = train_names[-round(train_num * self.val_size):]
				train_p_names = train_names[:len(train_names)-round(train_num * self.val_size)]
				val_num = len(val_p_names)
				train_num = len(train_p_names)
				test_p_names = index_df[(index_df['dataset']==da) & (index_df['set']=="testing")]['subj'].values
				test_num = len(test_p_names)
				# 构造 cfs-visit5-800002.edf 格式的文件名
				train_p_names = [f"cfs-visit5-{i}.edf" for i in train_p_names]
				val_p_names = [f"cfs-visit5-{i}.edf" for i in val_p_names]
				test_p_names = [f"cfs-visit5-{i}.edf" for i in test_p_names]

				print("train_num: ", train_num, "val_num: ", val_num, "test_num: ", test_num)
				tvt = {'train': train_p_names, 'val': val_p_names, 'test': test_p_names}

				for c, names in tvt.items():
					for (idx,name) in enumerate(names):
						raw_data = mne.io.read_raw_edf(os.path.join(data_dir,'edfs_new',name))
						print(os.path.join(data_dir,'edfs_new',name))
						raw_data.pick_channels(channels)
						xall = raw_data.get_data() # (2,n)
						# 降采样；128Hz->100Hz
						xall = signal.resample(xall, int(xall.shape[1]*100//128), axis=1)
						# 对时间维度规范化
						xall = (xall - np.mean(xall, axis=1, keepdims=True)) / np.std(xall, axis=1, keepdims=True)# (2,n)
						
						# 读取标签
						tree = ET.parse(os.path.join(data_dir,'annotations-events-profusion',name.split('.')[0]+'-profusion.xml'))
						root = tree.getroot()
						# 提取其中<SleepStage>标签
						labels = []
						for child in root:
							if child.tag == 'SleepStages':
								for stage in child:
									labels.append(int(stage.text))
						
						y_group = np.array(labels)
						# 将其中的4改为3
						y_group[y_group==4] = 3
						# 将其中的5改为4
						y_group[y_group==5] = 4
						# 如果存在不属于[0,1,2,3,4]的（x,y）删去，即6或9
						# 获取不属于[0,1,2,3,4]的index
						invalid_index = np.where(np.logical_not(np.isin(y_group, [0,1,2,3,4])))
						# 删除
						y_group = np.delete(y_group, invalid_index[0], axis=0)
						nums = xall.shape[1]//(sampling_rate*30)
						assert  nums == len(labels), f"epoch nums: {nums} != len(labels): {len(labels)}"
						xall = xall.reshape(2,nums,sampling_rate*30)
						xall = np.delete(xall, invalid_index[0], axis=1)

						# 划窗
						num_windows = xall.shape[1] // (self.input_epoch_num)
						# self.epoch_num_by_sub.append(num_windows*self.input_epoch_num) # 记录每个sub的标签数
						x_group = xall[:,:num_windows * self.input_epoch_num ,:].reshape(-1,num_windows,self.input_epoch_num * sampling_rate * 30).transpose(1,0,2) # (num_windows,2,120*3000)
						y_group = y_group[:num_windows * self.input_epoch_num].reshape((num_windows, self.input_epoch_num)) # (num_windows,input_epoch_num)

						x_group = torch.from_numpy(x_group).float()
						y_group = torch.from_numpy(y_group).float()

						if c=="test":
							if self.test_dataset == 'self' or self.test_dataset == 'cfs':
								data[c].append(x_group)
								label[c].append(y_group)
								if self.yasa_feature!="False":
									data_yasa[c].append(features)
							else:
								data['train'].append(x_group)
								label['train'].append(y_group)
								if self.yasa_feature!="False":
									data_yasa['train'].append(features)
						else:
							data[c].append(x_group)
							label[c].append(y_group)
							if self.yasa_feature!="False":
								data_yasa[c].append(features)
			elif d=="homepap":

				data_dir=datasets_dir[d]
				channels=channels_selected[d]
				p_names = [os.path.join(data_dir,'edfs_0602','lab','full',i) for i in os.listdir(os.path.join(data_dir, 'edfs_0602','lab','full'))]
				p_names.extend([os.path.join(data_dir,'edfs_0602','lab','split',i) for i in os.listdir(os.path.join(data_dir, 'edfs_0602','lab','split'))])
				p_names.sort()
				val_num = round(len(p_names) * self.val_size)
				test_num = round(len(p_names) * self.test_size)
				train_num = len(p_names) - val_num - test_num
				train_p_names = p_names[:train_num]
				val_p_names = p_names[train_num:train_num + val_num]
				test_p_names = p_names[train_num + val_num:]
				print("train_num: ", train_num, "val_num: ", val_num, "test_num: ", test_num)
				tvt = {'train': train_p_names, 'val': val_p_names, 'test': test_p_names}

				for c, names in tvt.items():
					for (idx,name) in enumerate(names):
						# if name.endswith('.edf'):
						# 	raw_data = mne.io.read_raw_edf(name)# 只有两个通道，不需要pick 250604现在需要了
							
						# 	# print(f"Reading file: {name}")
						# 	# print("Available channels:", raw_data.info['ch_names'])
						# 	# print("Requested channels:", channels)

						# 	# try:
						# 	# 	raw_data.pick_channels(channels)
						# 	# except Exception as e:
						# 	# 	print(f"Error picking channels from {name}: {e}")
						# 	# 	continue
							
						# 	raw_data.pick_channels(channels)
						# 	print(name)
						# 	freq = raw_data.info['sfreq']
						# 	xall = raw_data.get_data() # (2,n) # 256Hz
						# 	# 降采样；2**Hz->100Hz
						# 	xall = signal.resample(xall, int(xall.shape[1]*100//freq), axis=1)
						# 	# 构造label_file name
						# 	label_file = os.path.join(data_dir,'annotations-events-profusion','lab','split',name.split('/')[-1].split('.')[0]+'-profusion.xml')
						if name.endswith('.fif'):
							# print(f"Reading file: {name}")
							# print("Available channels:", raw_data.info['ch_names'])
							# print("Requested channels:", channels)

							raw_data = mne.io.read_raw_fif(name)
							raw_data.pick_channels(channels)
							print(name)
							freq = raw_data.info['sfreq']
							xall = raw_data.get_data() # (2,n) # 200Hz
							# 降采样；2**Hz->100Hz
							xall = signal.resample(xall, int(xall.shape[1]*100//freq), axis=1)
							# 构造label_file name
							label_file = os.path.join(data_dir,'annotations-events-profusion','lab',name.split('/')[-2],name.split('/')[-1].split('.')[0]+'-profusion.xml')
						# 对时间维度规范化
						xall = (xall - np.mean(xall, axis=1, keepdims=True)) / np.std(xall, axis=1, keepdims=True)# (2,n)
						# 如果只有一个通道，continue
						if xall.shape[0] == 1:
							continue
						
						# 读取标签
						tree = ET.parse(label_file)
						root = tree.getroot()
						# 提取其中<SleepStage>标签
						labels = []
						for child in root:
							if child.tag == 'SleepStages':
								for stage in child:
									labels.append(int(stage.text))
						
						y_group = np.array(labels) # 包含[0,1,2,3,4,5,6]
						# 将其中的4改为3
						y_group[y_group==4] = 3
						# 将其中的5改为4
						y_group[y_group==5] = 4
						# 如果存在不属于[0,1,2,3,4]的（x,y）删去，即6或9
						# 获取不属于[0,1,2,3,4]的index
						invalid_index = np.where(np.logical_not(np.isin(y_group, [0,1,2,3,4])))
						# 删除
						y_group = np.delete(y_group, invalid_index[0], axis=0)
						nums = xall.shape[1]//(sampling_rate*30)
						assert  nums == len(labels), f"epoch nums: {nums} != len(labels): {len(labels)}"
						# print(xall.shape)
						if xall.shape[1] % (sampling_rate*30) != 0: # 不是30s的整数倍,去掉最后一段
							xall = xall[:,:nums*sampling_rate*30]

						print(xall.shape)
						xall = xall.reshape(-1,nums,sampling_rate*30)
						print(xall.shape)
						xall = np.delete(xall, invalid_index[0], axis=1)

						# 划窗
						num_windows = xall.shape[1] // (self.input_epoch_num)
						# self.epoch_num_by_sub.append(num_windows*self.input_epoch_num) # 记录每个sub的标签数
						x_group = xall[:,:num_windows * self.input_epoch_num ,:].reshape(-1,num_windows,self.input_epoch_num * sampling_rate * 30).transpose(1,0,2) # (num_windows,2,120*3000)
						y_group = y_group[:num_windows * self.input_epoch_num].reshape((num_windows, self.input_epoch_num)) # (num_windows,input_epoch_num)

						x_group = torch.from_numpy(x_group).float()
						y_group = torch.from_numpy(y_group).float()

						if c=="test":
							if self.test_dataset == 'self' or self.test_dataset == 'cfs':
								data[c].append(x_group)
								label[c].append(y_group)
								if self.yasa_feature!="False":
									data_yasa[c].append(features)
							else:
								data['train'].append(x_group)
								label['train'].append(y_group)
								if self.yasa_feature!="False":
									data_yasa['train'].append(features)
						else:
							data[c].append(x_group)
							label[c].append(y_group)
							if self.yasa_feature!="False":
								data_yasa[c].append(features)
				if data_double:
					data_dir=datasets_dir[d]
					channels=channels_selected2[d]
					p_names = [os.path.join(data_dir,'edfs_0602','lab','full',i) for i in os.listdir(os.path.join(data_dir, 'edfs_0602','lab','full'))]
					p_names.extend([os.path.join(data_dir,'edfs_0602','lab','split',i) for i in os.listdir(os.path.join(data_dir, 'edfs_0602','lab','split'))])
					p_names.sort()
					val_num = round(len(p_names) * self.val_size)
					test_num = round(len(p_names) * self.test_size)
					train_num = len(p_names) - val_num - test_num
					train_p_names = p_names[:train_num]
					val_p_names = p_names[train_num:train_num + val_num]
					test_p_names = p_names[train_num + val_num:]
					print("train_num: ", train_num, "val_num: ", val_num, "test_num: ", test_num)
					tvt = {'train': train_p_names, 'val': val_p_names, 'test': test_p_names}
					for c, names in tvt.items():
						for (idx,name) in enumerate(names):
							# if name.endswith('.edf'):
							# 	raw_data = mne.io.read_raw_edf(name)# 只有两个通道，不需要pick 250604现在需要了
							# 	raw_data.pick_channels(channels)
							# 	print(name)
							# 	freq = raw_data.info['sfreq']
							# 	xall = raw_data.get_data() # (2,n) # 256Hz
							# 	# 降采样；2**Hz->100Hz
							# 	xall = signal.resample(xall, int(xall.shape[1]*100//freq), axis=1)
							# 	# 构造label_file name
							# 	label_file = os.path.join(data_dir,'annotations-events-profusion','lab','full',name.split('/')[-1].split('.')[0]+'-profusion.xml')
							if name.endswith('.fif'):
								raw_data = mne.io.read_raw_fif(name)
								print(name)
								raw_data.pick_channels(channels)
								freq = raw_data.info['sfreq']
								xall = raw_data.get_data() # (2,n) # 200Hz
								# 降采样；2**Hz->100Hz
								xall = signal.resample(xall, int(xall.shape[1]*100//freq), axis=1)
								# 构造label_file name
								label_file = os.path.join(data_dir,'annotations-events-profusion','lab',name.split('/')[-2],name.split('/')[-1].split('.')[0]+'-profusion.xml')
							# 对时间维度规范化
							xall = (xall - np.mean(xall, axis=1, keepdims=True)) / np.std(xall, axis=1, keepdims=True)# (2,n)
							# 如果只有一个通道，continue
							if xall.shape[0] == 1:
								continue
							
							# 读取标签
							tree = ET.parse(label_file)
							root = tree.getroot()
							# 提取其中<SleepStage>标签
							labels = []
							for child in root:
								if child.tag == 'SleepStages':
									for stage in child:
										labels.append(int(stage.text))
							
							y_group = np.array(labels) # 包含[0,1,2,3,4,5,6]
							# 将其中的4改为3
							y_group[y_group==4] = 3
							# 将其中的5改为4
							y_group[y_group==5] = 4
							# 如果存在不属于[0,1,2,3,4]的（x,y）删去，即6或9
							# 获取不属于[0,1,2,3,4]的index
							invalid_index = np.where(np.logical_not(np.isin(y_group, [0,1,2,3,4])))
							# 删除
							y_group = np.delete(y_group, invalid_index[0], axis=0)
							nums = xall.shape[1]//(sampling_rate*30)
							assert  nums == len(labels), f"epoch nums: {nums} != len(labels): {len(labels)}"
							# print(xall.shape)
							if xall.shape[1] % (sampling_rate*30) != 0: # 不是30s的整数倍,去掉最后一段
								xall = xall[:,:nums*sampling_rate*30]
							print(xall.shape)
							xall = xall.reshape(-1,nums,sampling_rate*30)
							xall = np.delete(xall, invalid_index[0], axis=1)

							# 划窗
							num_windows = xall.shape[1] // (self.input_epoch_num)
							# self.epoch_num_by_sub.append(num_windows*self.input_epoch_num) # 记录每个sub的标签数
							x_group = xall[:,:num_windows * self.input_epoch_num ,:].reshape(-1,num_windows,self.input_epoch_num * sampling_rate * 30).transpose(1,0,2) # (num_windows,2,120*3000)
							y_group = y_group[:num_windows * self.input_epoch_num].reshape((num_windows, self.input_epoch_num)) # (num_windows,input_epoch_num)

							x_group = torch.from_numpy(x_group).float()
							y_group = torch.from_numpy(y_group).float()

							if c=="test":
								if self.test_dataset == 'self' or self.test_dataset == 'cfs':
									data[c].append(x_group)
									label[c].append(y_group)
									if self.yasa_feature!="False":
										data_yasa[c].append(features)
								else:
									data['train'].append(x_group)
									label['train'].append(y_group)
									if self.yasa_feature!="False":
										data_yasa['train'].append(features)
							else:
								data[c].append(x_group)
								label[c].append(y_group)
								if self.yasa_feature!="False":
									data_yasa[c].append(features)
			elif d=="composite":
				# 2通道,100Hz,30s，(n,2,3000)(n,),已经分别规范化
				data_dir=datasets_dir[d]
				com_data = np.load(os.path.join(data_dir, 'data_x_y.npz'))
				x = com_data['arr_0']
				y = com_data['arr_1']
				# 划窗
				num_epochs = x.shape[0] // self.input_epoch_num
				x = x[:num_epochs * self.input_epoch_num].reshape((num_epochs, self.input_epoch_num, 2,-1)) # (num_windows,120,2,3000)
				x = x.transpose(0,2,1,3).reshape(num_epochs,2,-1) # (num_windows,2,120*3000)
				y = y[:num_epochs * self.input_epoch_num].reshape((num_epochs, self.input_epoch_num)) # (n',input_epoch_num)
				x = torch.from_numpy(x).float()
				y = torch.from_numpy(y).float()

				# 加入到train集
				data['train'].append(x)
				label['train'].append(y)
			elif d=="JXY_shhs1-300":
				data_dir=datasets_dir[d]
				channels=channels_selected[d]
				p_names = [i for i in os.listdir(data_dir)]
				p_names.sort()
				val_num = round(len(p_names) * self.val_size)
				test_num = round(len(p_names) * self.test_size)
				train_num = len(p_names) - val_num - test_num
				train_p_names = p_names[:train_num]
				val_p_names = p_names[train_num:train_num + val_num]
				test_p_names = p_names[train_num + val_num:]
				print("train_num: ", train_num, "val_num: ", val_num, "test_num: ", test_num)
				tvt = {'train': train_p_names, 'val': val_p_names, 'test': test_p_names}
				for c, names in tvt.items():
					for (idx,name) in enumerate(names):
						# 读取数据pkl
						with open(os.path.join(data_dir,name),'rb') as f:
							raw_data = pkl.load(f)
						xall = raw_data["data"] # (n,8,3000)
						# 规范化
						xall = (xall - np.mean(xall, axis=(0,2), keepdims=True)) / np.std(xall, axis=(0,2), keepdims=True)# (n,8,3000)
						# 读取标签
						if par_dict["task"] =="sleep_stage_classification":
							y = raw_data["sleep_labels"]
						elif par_dict["task"] == "sleep_apnea_detection":
							y = raw_data["apnea_labels"]
						elif par_dict["task"] == "sleep_arousal_detection":
							y = raw_data["arousal_labels"]
						# 多epoch划窗
						x_group,y_group=self.get_data_by_input_epoch_num(xall,y,self.input_epoch_num)
						if c=="test":
							if self.test_dataset == 'self' or self.test_dataset == 'JXY_shhs1-300':
								data[c].append(x_group)
								label[c].append(y_group)
							else:
								data['train'].append(x_group)
								label['train'].append(y_group)
						else:
							data[c].append(x_group)
							label[c].append(y_group)


		print("load data finish")

		x = data
		y = label
		x_yasa = None
		self.x_yasa = None
		if self.yasa_feature!="False":
			x_yasa = data_yasa
			x_yasa['train'] = torch.cat(x_yasa['train'], dim=0)
			x_yasa['val'] = torch.cat(x_yasa['val'], dim=0)
			x_yasa['test'] = torch.cat(x_yasa['test'], dim=0)
			self.x_yasa = x_yasa

		x['train'] = torch.cat(x['train'], dim=0)#.cuda()
		y['train'] = torch.cat(y['train'], dim=0)#.cuda()
		x['val'] = torch.cat(x['val'], dim=0)#.cuda()
		y['val'] = torch.cat(y['val'], dim=0)#.cuda()
		x['test'] = torch.cat(x['test'], dim=0)#.cuda()
		y['test'] = torch.cat(y['test'], dim=0)#.cuda()

		self.x = x
		self.y = y	


	def get_data_by_input_epoch_num(self,xall,stage_label,input_epoch_num):
		p = 0
		x_group = None
		y_group = None
		while p < len(stage_label):
			if p + input_epoch_num < len(stage_label):
				x = np.concatenate([xall[p+i,:,:] for i in range(input_epoch_num)],axis=1) # (4,3000*input_epoch_num)
				y = stage_label[p: p + input_epoch_num] # (input_epoch_num,)
			else:
				break  # 多出的一段不要了
			p += input_epoch_num

			if x_group is None:
				x_group = x[np.newaxis, :] # (1,4,3000*input_epoch_num)
				y_group = y[np.newaxis, :] # (1,input_epoch_num)
			else:
				x_group = np.append(x_group, x[np.newaxis, :], axis=0)
				y_group = np.append(y_group, y[np.newaxis, :], axis=0)

		# 转gpu
		x_group = torch.from_numpy(x_group).float() # (n',4,3000*input_epoch_num)
		y_group = torch.from_numpy(y_group).float() # (n',input_epoch_num)
		return x_group,y_group

	def gen_fold_data(self, fold):
		print(f"===============gen fold {fold} data with fold_num = {self.fold_num}===============")
		# 划分训练、验证、测试集的受试者id
		# self.fold_num 折跨人交叉验证,4/8折
		train_x,train_y = [],[]
		val_x,val_y = [],[]
		test_x,test_y = [],[]
		num=0
		print(type(self.x)) 
		for i in range(len(self.epoch_num_by_sub)):
			if i % self.fold_num ==fold:
				test_x.append(self.x[num:num+self.epoch_num_by_sub[i]//self.input_epoch_num])
				test_y.append(self.y[num:num+self.epoch_num_by_sub[i]//self.input_epoch_num])
			elif i % self.fold_num == (fold+1)%self.fold_num:
				val_x.append(self.x[num:num+self.epoch_num_by_sub[i]//self.input_epoch_num])
				val_y.append(self.y[num:num+self.epoch_num_by_sub[i]//self.input_epoch_num])
			else:
				train_x.append(self.x[num:num+self.epoch_num_by_sub[i]//self.input_epoch_num])
				train_y.append(self.y[num:num+self.epoch_num_by_sub[i]//self.input_epoch_num])
			num+=self.epoch_num_by_sub[i]//self.input_epoch_num

		train_x = torch.cat(train_x, dim=0)
		train_y = torch.cat(train_y, dim=0)
		val_x = torch.cat(val_x, dim=0)
		val_y = torch.cat(val_y, dim=0)
		test_x = torch.cat(test_x, dim=0)
		test_y = torch.cat(test_y, dim=0)
		
		train_dataset = MyDataset(train_x, train_y)
		val_dataset = MyDataset(val_x, val_y)
		test_dataset = MyDataset(test_x, test_y)

		print("train data shape: ", len(train_x))
		print("val data shape: ", len(val_x))
		print("test data shape: ", len(test_x))

		return train_dataset, val_dataset, test_dataset
	
	def gen_data(self):
		print(f"===============gen train:val:test data===============")

		# 已经打乱地划分了数据集

		train_x = self.x['train']
		val_x = self.x['val']
		test_x = self.x['test']

		train_y = self.y['train']
		val_y = self.y['val']
		test_y = self.y['test']

		train_y_np = train_y.numpy() if torch.is_tensor(train_y) else train_y
		train_y_smoothed = smooth_labels(train_y_np, num_classes=5, smoothing=0.1)
		train_y_tensor = torch.from_numpy(train_y_smoothed).float()

		if self.yasa_feature!="False":
			train_x_yasa = self.x_yasa['train']
			val_x_yasa = self.x_yasa['val']
			test_x_yasa = self.x_yasa['test']
			train_dataset = MyDataset_yasa(train_x, train_x_yasa, train_y)
			val_dataset = MyDataset_yasa(val_x, val_x_yasa, val_y)
			test_dataset = MyDataset_yasa(test_x, test_x_yasa, test_y)
			print("train data shape: ", len(train_x))
			print("val data shape: ", len(val_x))
			print("test data shape: ", len(test_x))
			
			return train_dataset, val_dataset, test_dataset

		# train_dataset = MyDataset(train_x, train_y)
		train_dataset = MyDataset(train_x, train_y_tensor)
		val_dataset = MyDataset(val_x, val_y)
		test_dataset = MyDataset(test_x, test_y)

		print("train data shape: ", len(train_x))
		print("val data shape: ", len(val_x))
		print("test data shape: ", len(test_x))

		return train_dataset, val_dataset, test_dataset
	
	def gen_testdata(self,dataset='taiyang'):
		print(f"===============gen test data only===============")
		test_x,test_y = None,None
		if dataset == "taiyang":
			test_x = self.x
			test_y = self.y
		else:
			test_x = self.x['test']
			test_y = self.y['test']

		test_dataset = MyDataset(test_x, test_y)

		print("test data shape: ", len(test_x))

		return test_dataset
