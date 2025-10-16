import json
import os
import random
import mne
# import neurokit2 as nk
import numpy as np
from xml.etree import ElementTree as ET
import numpy as np
import torch
from scipy import interpolate
import scipy.io as sio
from scipy import signal
from my_dataset import MyDatasetThreePre, MyDataset, MyDatasetOnePre
# from preprocess.loader import load_shhs_xml_stage_label, load_shhs_edf
import pickle as pkl
import pandas as pd

label_mapping = {
    'W': 0,
    'N1': 1,
    'N2': 2,
    'N3': 3,
    'REM': 4
}


class DataPreprocesserSHHS_test:
    def __init__(self, datasets_dir, channels_selected, test_dataset, input_epoch_num, sampling_rate, test_size):
        # self.fold_num = fold_num
        self.test_size = test_size
        # self.val_size = val_size
        self.test_dataset = test_dataset

        data_x = []
        # data['train'], data['val'], data['test'] = [], [], []
        label_y = []
        # label['train'], label['val'], label['test'] = [], [], []
        for d in test_dataset:
            if d == 'ISRUC1':
                data_dir = datasets_dir[d]
                channels = channels_selected[d]
                # 读取data_dir中所有以.mat结尾的文件
                p_names = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
                # 打乱顺序
                # random.shuffle(p_names)
                # 按8：1：1划分数据集
                p_names.sort()
                test_num = round(len(p_names)*self.test_size)
                p_names = p_names[-test_num:]
                for name in p_names:
                    raw_data = sio.loadmat(os.path.join(data_dir, name))
                    print(os.path.join(data_dir, name))
                    # 将所选通道的数据拼接
                    xall = np.concatenate([raw_data[i][:, np.newaxis, :] for i in channels], axis=1)  # (n,4,6000)
                    # 降采样；200Hz->100Hz
                    xall = signal.resample(xall, int(xall.shape[2] // 2), axis=2)  # (n,4,3000)
                    # 对每个窗口的每个通道进行规范化
                    xall = (xall - np.mean(xall, axis=(0,2), keepdims=True)) / np.std(xall, axis=(0,2), keepdims=True)# (n,4,3000)
                    index = name.split('.')[0][7:]  # 获取文件名中的index,用于获取标签
                    label_name = index + '_1.npy'
                    stage_label = np.load(os.path.join(data_dir, 'label', label_name))  # (n,)
                    x_group, y_group = self.get_data_by_input_epoch_num(xall, stage_label, input_epoch_num)
                    data_x.append(x_group)
                    label_y.append(y_group)

            elif d == "ISRUC3":
                data_dir = datasets_dir[d]
                channels = channels_selected[d]
                # 读取data_dir中所有以.mat结尾的文件
                p_names = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
                p_names.sort()
                test_num = round(len(p_names)*self.test_size)
                p_names = p_names[-test_num:]
                for name in p_names:
                    # if name == 'subject4.mat':
                    #     raw_data = sio.loadmat(os.path.join('/home/ShareData/', name))
                    # else:
                    #     raw_data = sio.loadmat(os.path.join(data_dir, name))
                    raw_data = sio.loadmat(os.path.join(data_dir, name))
                    # print(os.path.join(data_dir, name))
                    # 将所选通道的数据拼接
                    xall = np.concatenate([raw_data[i][:, np.newaxis, :] for i in channels], axis=1)  # (n,4,6000)
                    # 降采样；200Hz->100Hz
                    xall = signal.resample(xall, int(xall.shape[2] // 2), axis=2)  # (n,4,3000)
                    # 对每个窗口的每个通道进行规范化
                    xall = (xall - np.mean(xall, axis=(0,2), keepdims=True)) / np.std(xall, axis=(0,2), keepdims=True)# (n,4,3000)
                    index = name.split('.')[0][7:]  # 获取文件名中的index,用于获取标签
                    label_name = index + '-Label.mat'
                    stage_label = sio.loadmat(os.path.join(data_dir, 'label', label_name))['label'].reshape(
                        -1)  # (n,)
                    x_group, y_group = self.get_data_by_input_epoch_num(xall, stage_label, input_epoch_num)
                    data_x.append(x_group)
                    label_y.append(y_group)

            elif d == "MASS":
                data_dir = datasets_dir[d]
                channels = channels_selected[d]
                channel = ['FP1', 'FP2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'C3', 'C4', 'T3', 'T4', 'Pz', 'P3', 'P4', 'T5',
                           'T6', 'Oz', 'O1', 'O2', 'EogL', 'EogR', 'Emg1', 'Emg2', 'Emg3', 'Ecg']
                channel_index = [channel.index(c) for c in channels]
                # 读取data_dir中所有以-Datasub.mat结尾的文件
                p_names = [f for f in os.listdir(data_dir) if f.endswith('-Datasub.mat')]
                p_names.sort()
                test_num = round(len(p_names)*self.test_size)
                p_names = p_names[-test_num:]
                for name in p_names:
                    raw_data = sio.loadmat(os.path.join(data_dir, name))
                    print(os.path.join(data_dir, name))
                    # 将所选通道的数据拼接
                    xall = raw_data['PSG'][:, channel_index, :]  # (n,4,6000)
                    # 对时间维度规范化
                    xall = (xall - np.mean(xall, axis=(0,2), keepdims=True)) / np.std(xall, axis=(0,2), keepdims=True)# (n,4,3000)

                    index = name[:10]  # 获取文件名中的index,用于获取标签
                    label_name = index + '-Label.mat'
                    stage_label = sio.loadmat(os.path.join(data_dir, label_name))['label']  # (n,5)
                    stage_label = np.argmax(stage_label, axis=1)  # (n,)
                    x_group, y_group = self.get_data_by_input_epoch_num(xall, stage_label, input_epoch_num)
                    data_x.append(x_group)
                    label_y.append(y_group)

            elif d == "SHHS":
                data_dir = datasets_dir[d]
                channels = channels_selected[d]
                channel = ['EEG', "EEG(sec)", 'EOG(L)', 'EMG']
                channel_index = [channel.index(c) for c in channels]
                # 读取data_dir中所有以-Datasub.mat结尾的文件
                p_names = [f for f in os.listdir(data_dir)][1:501]
                p_names.sort()
                test_num = round(len(p_names)*self.test_size)
                p_names = p_names[-test_num:]
                for name in p_names:
                    print(os.path.join(data_dir, name))
                    if not os.path.exists(os.path.join(data_dir, name)):
                        print(f"The file {os.path.join(data_dir, name)} does not exist.")
                        continue
                    with open(os.path.join(data_dir, name), 'rb') as f:
                        try:
                            dic = pkl.load(f, encoding='latin1')
                        except pkl.UnpicklingError as e:
                            print(f"UnpicklingError: {e}")
                            continue
                        except Exception as e:
                            print(f"Failed to load data: {e}")
                            continue
                    # print(os.path.join(data_dir, name))
                    xall = dic["new_xall"][:, channel_index]
                    stage_label = dic["stage_label"]
                    # 对时间维度规范化
                    xall = (xall - np.mean(xall, axis=0, keepdims=True)) / np.std(xall, axis=0, keepdims=True)# (n,4)

                    # 按睡眠epoch划分
                    num_epochs = xall.shape[0] // (sampling_rate * 30)
                    xall = xall[:num_epochs * sampling_rate * 30].reshape(
                        (num_epochs, sampling_rate * 30, -1)
                    ).transpose(0, 2, 1)
                    stage_label = stage_label[:num_epochs]
                    x_group, y_group = self.get_data_by_input_epoch_num(xall, stage_label, input_epoch_num)
                    data_x.append(x_group)
                    label_y.append(y_group)

            elif d == "SLEEPEDF153":
                data_dir = datasets_dir[d]
                channels = channels_selected[d]
                channel = ['Fpz-Cz','EOG','EMG']
                channel_index = [channel.index(c) for c in channels]

                p_names = [f for f in os.listdir(data_dir)]
                p_names.sort()
                test_num = round(len(p_names)*self.test_size)
                p_names = p_names[-test_num:]
                for name in p_names:
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
                    # 规范化
                    x = (x - np.mean(x, axis=(0,2), keepdims=True)) / np.std(x, axis=(0,2), keepdims=True)# (n,4,3000)
                    y = npz_file['y']

                    # print(x.shape, y.shape)
                    x_group, y_group = self.get_data_by_input_epoch_num(x, y, input_epoch_num)
                    data_x.append(x_group)
                    label_y.append(y_group)

            elif d == "TaiYang":
                data_dir = datasets_dir[d]
                channels = channels_selected[d]
                # channel=['EEG',"EEG(sec)", 'EOG(L)', 'EMG']
                # channel_index=[channel.index(c) for c in channels]

                p_names = [f for f in os.listdir(data_dir)]
                # 打乱顺序
                # random.shuffle(p_names)
                # 为文件名字排序
                p_names.sort()
                all_data = []
                all_label = []
                for subject_id, subject_folder in enumerate(p_names):
                    subject_folder = os.path.join(data_dir, subject_folder)

                    if os.path.isdir(subject_folder):
                        edf_file = None
                        json_file = None

                        for file in os.listdir(subject_folder):
                            if file.endswith('.edf'):
                                edf_file = os.path.join(subject_folder, file)
                            elif file.endswith('.json'):
                                json_file = os.path.join(subject_folder, file)

                        if edf_file and json_file:
                            print(f'Processing subject {subject_id}')
                            data = self.load_shhs_edf(edf_file, channels).to_numpy()

                            labels = self.load_json_labels(json_file)
                            labels = np.array(labels)
                            num_epochs = data.shape[0] // (sampling_rate * 30)
                            data = data[:num_epochs * sampling_rate * 30].reshape(
                                (num_epochs, sampling_rate * 30, -1)
                            ).transpose(0, 2, 1)
                            labels = labels[:num_epochs]
                            data, labels = self.get_data_by_input_epoch_num(data, labels, input_epoch_num)
                            # 确保 data 和 labels 是 NumPy 数组
                            data = np.array(data)
                            labels = np.array(labels)
                    if data is not None and labels is not None:
                        # 将它们添加到列表中
                        data_x.append(data)
                        label_y.append(labels)
                    else:
                        print("data 或 labels 是 None")
                        continue
                    # all_data=all_data.append(np.array(data))
                    # all_label=all_label.append(np.array(labels))
                    # print([d.shape for d in all_data])
                # all_data = np.concatenate(all_data, axis=0)
                # all_label = np.concatenate(all_label, axis=0)
        # print(data_x.shape)
        
        # for i, d in enumerate(data_x):
        #     print(f"Shape of data_x[{i}]: {d.shape}")
        data_x = np.concatenate(data_x, axis=0)
        label_y = np.concatenate(label_y, axis=0)
        print(data_x.shape)
        print("load data finish")

        x = torch.from_numpy(data_x).float()
        y = torch.from_numpy(label_y).float()

        # x['train'] = torch.cat(x['train'], dim=0)#.cuda()
        # y['train'] = torch.cat(y['train'], dim=0)#.cuda()
        # x['val'] = torch.cat(x['val'], dim=0)#.cuda()
        # y['val'] = torch.cat(y['val'], dim=0)#.cuda()
        # x['test'] = torch.cat(x['test'], dim=0)#.cuda()
        # y['test'] = torch.cat(y['test'], dim=0)#.cuda()

        self.x = x
        self.y = y

    def get_data_by_input_epoch_num(self, xall, stage_label, input_epoch_num):
        p = 0
        x_group = None
        y_group = None
        while p < len(stage_label):
            if p + input_epoch_num < len(stage_label):
                x = np.concatenate([xall[p + i, :, :] for i in range(input_epoch_num)],
                                   axis=1)  # (4,3000*input_epoch_num)
                y = stage_label[p: p + input_epoch_num]  # (input_epoch_num,)
            else:
                break  # 多出的一段不要了
            p += input_epoch_num

            if x_group is None:
                x_group = x[np.newaxis, :]  # (1,4,3000*input_epoch_num)
                y_group = y[np.newaxis, :]  # (1,input_epoch_num)
            else:
                x_group = np.append(x_group, x[np.newaxis, :], axis=0)
                y_group = np.append(y_group, y[np.newaxis, :], axis=0)

        # 转gpu
        # x_group = torch.from_numpy(x_group).float() # (n',4,3000*input_epoch_num)
        # y_group = torch.from_numpy(y_group).float() # (n',input_epoch_num)
        return x_group, y_group

    def load_shhs_edf(self, data_dir, channels):
        raw_data = mne.io.read_raw_edf(data_dir)
        # required_channels = ['EEG Fp1-REF', 'EEG C4-REF', "EOG1",'EMG'] #通道
        required_channels = channels
        available_channels = raw_data.ch_names
        channels_to_pick = [ch for ch in required_channels if ch in available_channels]

        if len(channels_to_pick) != len(required_channels):
            missing_channels = set(required_channels) - set(channels_to_pick)
            raise ValueError(f"Missing required channels: {missing_channels}")

        raw_data.pick(required_channels)
        raw_dataframe = raw_data.to_data_frame().drop(columns=['time'])

        return raw_dataframe

    def load_json_labels(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            df = pd.DataFrame(data)
            df = df['label_name'].replace(label_mapping).to_numpy()
        # print(df)
        return df

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
        train_val_x = torch.cat((self.x[0:test_cut_begin], self.x[test_cut_end:]), dim=0)
        val_cut_begin = int(len(train_val_x) * train_val_cut_rate)
        train_x = train_val_x[0:val_cut_begin]
        val_x = train_val_x[val_cut_begin:]
        test_x = self.x[test_cut_begin: test_cut_end]

        train_val_y = torch.cat((self.y[0:test_cut_begin], self.y[test_cut_end:]), dim=0)
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

        # # 把人数展平
        # train_x = torch.cat(train_x, dim=0)
        # train_y = torch.cat(train_y, dim=0)
        # val_x = torch.cat(val_x, dim=0)
        # val_y = torch.cat(val_y, dim=0)
        # test_x = torch.cat(test_x, dim=0)
        # test_y = torch.cat(test_y, dim=0)

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

    def gen_data(self):
        print(f"===============gen train:val:test data===============")

        # 已经打乱地划分了数据集

        # train_x = self.x['train']
        # val_x = self.x['val']
        test_x = self.x

        # train_y = self.y['train']
        # val_y = self.y['val']
        test_y = self.y

        # train_dataset = MyDataset(train_x, train_y)
        # val_dataset = MyDataset(val_x, val_y)
        test_dataset = MyDataset(test_x, test_y)

        # print("train data shape: ", len(train_x))
        # print("val data shape: ", len(val_x))
        print("test data shape: ", len(test_x))

        return test_dataset


if __name__ == '__main__':
    dataset_dir = {}
    dataset_dir['TaiYang'] = "/home/ShareData/SunData"
    dataset_dir['ISRUC1'] = "/home/ShareData/ISRUC-1/ISRUC-1"
    channels_selected = {}
    channels_selected['TaiYang'] = ['EEG C4-REF', 'EEG Fp2-REF', 'EOG1', 'EMG']
    channels_selected['ISRUC1'] =  ["F4_A1","C4_A1","ROC_A1","X1"]
    DataPreprocesserSHHS_test(dataset_dir, channels_selected, ["ISRUC1"], 20, 100)