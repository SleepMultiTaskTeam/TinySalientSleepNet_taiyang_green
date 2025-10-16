import os
import random

import neurokit2 as nk
import numpy as np
from xml.etree import ElementTree as ET
import numpy as np
import torch
from scipy import interpolate
import scipy.io as sio
from scipy import signal
from my_dataset import MyDatasetThreePre, MyDataset, MyDatasetOnePre
from preprocess.loader import load_shhs_xml_stage_label, load_shhs_edf
import pickle as pkl


class DataPreprocesserSHHS:
	def __init__(self, datasets_dir, channels_selected, dataset, test_dataset, input_epoch_num, sampling_rate, fold_num,test_size,val_size):
		self.fold_num = fold_num
		self.test_size = test_size
		self.val_size = val_size
		self.test_dataset = test_dataset

		data = {}
		data['train'], data['val'], data['test']=[],[],[]
		label = {}
		label['train'], label['val'], label['test']=[],[],[]
		for d in dataset:
			if d=='ISRUC1':
				data_dir=datasets_dir[d]
				channels=channels_selected[d]
				# 读取data_dir中所有以.mat结尾的文件
				p_names = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
				# 打乱顺序
				random.shuffle(p_names)
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
						index=name.split('.')[0][7:] # 获取文件名中的index,用于获取标签
						label_name=index+'_1.npy'
						stage_label=np.load(os.path.join(data_dir,'label',label_name)) # (n,)
						x_group,y_group=self.get_data_by_input_epoch_num(xall,stage_label,input_epoch_num)

						if c=="test":
							if self.test_dataset == 'self' or self.test_dataset == 'ISRUC':
								data[c].append(x_group)
								label[c].append(y_group)
							else:
								data['train'].append(x_group)
								label['train'].append(y_group)
						else:
							data[c].append(x_group)
							label[c].append(y_group)
			elif d=="ISRUC3":
				data_dir=datasets_dir[d]
				channels=channels_selected[d]
				# 读取data_dir中所有以.mat结尾的文件
				p_names = [f for f in os.listdir(data_dir) if f.endswith('.mat') ]
				# 打乱顺序
				random.shuffle(p_names)
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
						print(os.path.join(data_dir, name))
						# if name == 'subject4.mat':
						# 	raw_data = sio.loadmat(os.path.join('/home/ShareData/', name))
						# else:
						raw_data = sio.loadmat(os.path.join(data_dir, name))
						# print(os.path.join(data_dir, name))
						# 将所选通道的数据拼接
						xall = np.concatenate([raw_data[i][:,np.newaxis,:] for i in channels],axis=1) # (n,4,6000)
						# 降采样；200Hz->100Hz
						xall = signal.resample(xall, int(xall.shape[2]//2), axis=2) # (n,4,3000)
						index=name.split('.')[0][7:] # 获取文件名中的index,用于获取标签
						label_name=index+'-Label.mat'
						stage_label=sio.loadmat(os.path.join(data_dir,'label',label_name))['label'].reshape(-1) # (n,)
						# print(xall.shape, stage_label.shape)
						x_group,y_group=self.get_data_by_input_epoch_num(xall,stage_label,input_epoch_num)
						# print(x_group.shape, y_group.shape)
						if c=="test":
							if self.test_dataset == 'self' or self.test_dataset == 'ISRUC':
								data[c].append(x_group)
								label[c].append(y_group)
							else:
								data['train'].append(x_group)
								label['train'].append(y_group)
						else:
							data[c].append(x_group)
							label[c].append(y_group)
			elif d=="MASS":
				data_dir=datasets_dir[d]
				channels=channels_selected[d]
				channel=['FP1','FP2','Fz' , 'F3' , 'F4' , 'F7' , 'F8' , 'C3' , 'C4' , 'T3' , 'T4' , 'Pz' , 'P3' , 'P4' , 'T5' , 'T6' , 'Oz' , 'O1' , 'O2' , 'EogL' , 'EogR' , 'Emg1' , 'Emg2' , 'Emg3' , 'Ecg' ]
				channel_index=[channel.index(c) for c in channels]
				# 读取data_dir中所有以-Datasub.mat结尾的文件
				p_names = [f for f in os.listdir(data_dir) if f.endswith('-Datasub.mat')]
				# 打乱顺序
				random.shuffle(p_names)
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
						xall = raw_data['PSG'][:,channel_index,:] # (n,4,6000)
						
						index=name[:10] # 获取文件名中的index,用于获取标签
						label_name=index+'-Label.mat'
						stage_label=sio.loadmat(os.path.join(data_dir,label_name))['label'] # (n,5)
						stage_label = np.argmax(stage_label, axis=1) # (n,)
						x_group,y_group=self.get_data_by_input_epoch_num(xall,stage_label,input_epoch_num)

						if c=="test":
							if self.test_dataset == 'self' or self.test_dataset == 'MASS':
								data[c].append(x_group)
								label[c].append(y_group)
							else:
								data['train'].append(x_group)
								label['train'].append(y_group)
						else:
							data[c].append(x_group)
							label[c].append(y_group)
			elif d=="SHHS":
				data_dir=datasets_dir[d]
				channels=channels_selected[d]
				channel=['EEG',"EEG(sec)", 'EOG(L)', 'EMG']
				channel_index=[channel.index(c) for c in channels]
				# 读取data_dir中所有以-Datasub.mat结尾的文件
				p_names = [f for f in os.listdir(data_dir)][1:501]
				# 打乱顺序
				random.shuffle(p_names)
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
				# 	with open(os.path.join(shhs1_process1_dir, name), 'rb') as f:
                #     dic = pkl.load(f)
                # x_group = dic["new_xall"][:, [0, 2]]
                # y_group = dic["stage_label"]
					for name in names:
						print(os.path.join(data_dir, name))
						if not os.path.exists(os.path.join(data_dir, name)):
							print(f"The file {os.path.join(data_dir, name)} does not exist.")
							continue
						with open(os.path.join(data_dir, name), 'rb') as f:
							try:
								dic = pkl.load(f,encoding='latin1')
							except pkl.UnpicklingError as e:
								print(f"UnpicklingError: {e}")
								continue
							except Exception as e:
								print(f"Failed to load data: {e}")
								continue
						# print(os.path.join(data_dir, name))
						xall = dic["new_xall"][:,channel_index]
						stage_label = dic["stage_label"]	
      
						# 按睡眠epoch划分
						num_epochs = xall.shape[0] // (sampling_rate * 30)
						xall = xall[:num_epochs * sampling_rate * 30].reshape(
                    	(num_epochs, sampling_rate * 30, -1)
                		).transpose(0,2,1)
						stage_label = stage_label[:num_epochs]
						# 将所选通道的数据拼接
						# xall = raw_data['PSG'][:,channel_index,:] # (n,4,6000)
						
						# index=name[:10] # 获取文件名中的index,用于获取标签
						# label_name=index+'-Label.mat'
						# stage_label=sio.loadmat(os.path.join(data_dir,label_name))['label'] # (n,5)
						# stage_label = np.argmax(stage_label, axis=1) # (n,)
						x_group,y_group=self.get_data_by_input_epoch_num(xall,stage_label,input_epoch_num)

						if c=="test":
							if self.test_dataset == 'self' or self.test_dataset == 'SHHS':
								data[c].append(x_group)
								label[c].append(y_group)
							else:
								data['train'].append(x_group)
								label['train'].append(y_group)
						else:
							data[c].append(x_group)
							label[c].append(y_group)

			elif d == "SLEEPEDF153":
				data_dir = datasets_dir[d]
				channels = channels_selected[d]
				channel = ['Fpz-Cz', "PZz-Oz", 'EOG', 'EMG']
				channel_index = [channel.index(c) for c in channels]
				
				p_names = [f for f in os.listdir(data_dir)]
				random.shuffle(p_names)
				val_num = round(len(p_names) * self.val_size)
				test_num = round(len(p_names) * self.test_size)
				train_num = len(p_names) - val_num - test_num
				train_p_names = p_names[:train_num]
				val_p_names = p_names[train_num:train_num + val_num]
				test_p_names = p_names[train_num + val_num:]
				print("train_num: ", train_num, "val_num: ", val_num, "test_num: ", test_num)
				tvt = {'train': train_p_names, 'val': val_p_names, 'test': test_p_names}

				# Initialize lists for data and label
				data = {'train': [], 'val': [], 'test': []}
				label = {'train': [], 'val': [], 'test': []}

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

						x = npz_file['x'][:, :, channel_index].transpose(0, 2, 1)
						y = npz_file['y']

						# print(x.shape, y.shape)
						x_group,y_group=self.get_data_by_input_epoch_num(x,y,input_epoch_num)
						# print(x_group.shape, y_group.shape)

						if c == "test":
							if self.test_dataset == 'self' or self.test_dataset == 'SLEEPEDF153':
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

	
	
	def gen_data(self):
		print(f"===============gen train:val:test data===============")

		# 已经打乱地划分了数据集

		train_x = self.x['train']
		val_x = self.x['val']
		test_x = self.x['test']

		train_y = self.y['train']
		val_y = self.y['val']
		test_y = self.y['test']


		train_dataset = MyDataset(train_x, train_y)
		val_dataset = MyDataset(val_x, val_y)
		test_dataset = MyDataset(test_x, test_y)

		print("train data shape: ", len(train_x))
		print("val data shape: ", len(val_x))
		print("test data shape: ", len(test_x))

		return train_dataset, val_dataset, test_dataset
