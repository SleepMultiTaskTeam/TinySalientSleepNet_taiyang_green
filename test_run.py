import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from preprocess.data_preprocesser_taiyang import DataPreprocesser
from torch.utils.data import DataLoader
from model.origin_SalientSleepNet import TinySalientSleepNet
import logging
import argparse
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import json
import matplotlib.pyplot as plt
import pickle as pkl

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s]%(message)s",
        datefmt='%Y.%m.%d. %H:%M:%S'
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def get_args():
    parser = argparse.ArgumentParser('Train')

    # General args
    # data loading
    parser.add_argument('--data_path',      type=str, default="/mnt/nfsData17/ZhaiMengjie/Datasets/taiyang_2/")
    parser.add_argument('--model_path',      type=str, default="./result/tinysalient/test_doubledata/")
    # data loading
    parser.add_argument('--dataset',      type=str, default="taiyang")
    # save folder name 
    parser.add_argument('--folder_name',      type=str, default="result")
    # log name 
    parser.add_argument('--log_name',      type=str, default="test_run.log")
    # batch_size
    parser.add_argument('--batch_size',      type=int, default=8)
    # input epoch num
    parser.add_argument('--input_epoch_num',      type=int, default=120)
    # sampling rate
    parser.add_argument('--sampling_rate',      type=int, default=100)
    # channels
    parser.add_argument('--channels',      type=str, default="EEG C4-REF,EEG O1-REF,EEG T4-REF,EOG1")#"EEG C4-REF,EOG1") EEG O2-REF
    # seed
    parser.add_argument('--seed',      type=int, default=42)
    # gpu
    parser.add_argument('--gpu',      type=str, default="3")
    # label availabel
    parser.add_argument('--label_available',      action='store_true', default=False)
    parser.add_argument('--ICA',      action='store_true', default=False)
    parser.add_argument('--real_time',      action='store_true', default=False)
    parser.add_argument('--vote',      action='store_true', default=False)
    
    args = parser.parse_args()
    args.channels = args.channels.split(",")

    return args

def set_seed(args):
    # 固定种子
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def test_result(net, test_loader, window_num_by_sub=None, last_epoch_num_by_sub=None, device=None,args=None):
    # test
    ypred = []
    ytrue = []
    ypred_sub_list = []
    ytrue_sub_list = []
    ypred_softmax_list = []
    net.eval()
    test_all_time = 0
    for i, (batch_x, batch_y) in enumerate(test_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device) # (batch, input_epoch_num)
        start_time = time.time()
        y_pre = net(batch_x) # (batch,5,input_epoch_num)
        test_time = time.time() - start_time
        test_all_time += test_time
        y_pre = F.softmax(y_pre, dim=1)
        if args.real_time and not args.vote:
            # 实时预测，直接取最后一个时间步的预测结果
            y_pre = y_pre[:, :, -1]
            batch_y = batch_y[:, -1]
        elif args.real_time and args.vote:
            batch_y = batch_y[:, -1]
            # 实时预测，投票机制，前面部分的每个epoch都有120个预测结果，先记录下来，再求平均
            ypred_softmax_list.append(y_pre.detach().cpu().numpy())
        y_pre = y_pre.argmax(1)
        ypred.append(y_pre.cpu().numpy())
        ytrue.append(batch_y.cpu().numpy())
    logger.info(f"test_all_time: {test_all_time}")
    ypred, ytrue = np.concatenate(ypred, axis=0), np.concatenate(ytrue, axis=0)
    if args.real_time and args.vote:
        ypred_softmax_list = np.concatenate(ypred_softmax_list, axis=0) # (allB,5,120)
    # 计算每个子被试的准确率
    for i in range(len(window_num_by_sub)):
        if last_epoch_num_by_sub[i] > 0:
            ypred_sub = ypred[sum(window_num_by_sub[:i]) : sum(window_num_by_sub[:i+1])-1].reshape(-1)
            ypred_last = ypred[sum(window_num_by_sub[:i+1])-1].reshape(-1)[-last_epoch_num_by_sub[i]:]
            ypred_sub = np.concatenate((ypred_sub, ypred_last), axis=0)
            ypred_sub_list.append(ypred_sub)
            ytrue_sub = ytrue[sum(window_num_by_sub[:i]) : sum(window_num_by_sub[:i+1])-1].reshape(-1)
            ytrue_last = ytrue[sum(window_num_by_sub[:i+1])-1].reshape(-1)[-last_epoch_num_by_sub[i]:]
            ytrue_sub = np.concatenate((ytrue_sub, ytrue_last), axis=0)
            ytrue_sub_list.append(ytrue_sub)
        else:
            if args.vote:
                # 从第120列开始进行反对角线求平均
                ypred_softmax = ypred_softmax_list[sum(window_num_by_sub[:i]) : sum(window_num_by_sub[:i+1])]
                ypred_softmax = np.transpose(ypred_softmax, (0, 2, 1))  # (num_windows, input_epoch_num, 5)

                num_windows, input_epoch_num, num_classes = ypred_softmax.shape
                total_epochs = num_windows

                vote_sum = np.zeros((total_epochs, num_classes))
                vote_count = np.zeros((total_epochs,))

                for w in range(num_windows):
                    for e in range(input_epoch_num):
                        epoch_idx = w - (input_epoch_num - 1 - e)
                        if 0 <= epoch_idx < total_epochs:
                            vote_sum[epoch_idx] += ypred_softmax[w, e]
                            vote_count[epoch_idx] += 1

                avg_softmax = vote_sum / vote_count[:, None]
                ypred_sub = np.argmax(avg_softmax, axis=1)
                ytrue_sub = ytrue[sum(window_num_by_sub[:i]) : sum(window_num_by_sub[:i+1])].reshape(-1)
                ypred_sub_list.append(ypred_sub)
                ytrue_sub_list.append(ytrue_sub)                
            else:
                ypred_sub = ypred[sum(window_num_by_sub[:i]) : sum(window_num_by_sub[:i+1])].reshape(-1)
                ypred_sub_list.append(ypred_sub)
                ytrue_sub = ytrue[sum(window_num_by_sub[:i]) : sum(window_num_by_sub[:i+1])].reshape(-1)
                ytrue_sub_list.append(ytrue_sub)

    return ypred_sub_list, ytrue_sub_list

# 修改处 1：添加 label 提取逻辑
def plot_and_print_cm(ypre, y_true, cm_out_dir, label_class=None, file_name=""):
    labels = np.union1d(y_true, ypre)  # union to cover all classes
    cm = confusion_matrix(y_true, ypre, labels=labels)
    cm = np.array(cm)

    # ✅ 修改：如果 label_class 是原始映射表（如 wake/N1...），从中筛出当前标签名
    if label_class and len(label_class) > max(labels):
        label_names = [label_class[int(i)] for i in labels]
    else:
        label_names = [str(i) for i in labels]

    plot_confusion_matrix(cm, classes=label_names, title=file_name + 'cm_', path=cm_out_dir, normalize=True)
    plot_confusion_matrix(cm, classes=label_names, title=file_name + 'cm_num', normalize=False, path=cm_out_dir)

    return cm, None


# 修改处 3：重写归一化逻辑，防止除以0，并传入 labels
def plot_confusion_matrix(cm: np.ndarray, classes: list,
                          normalize: bool = True, title: str = None,
                          cmap: str = 'Blues', path: str = ''):
    if not title:
        title = 'Normalized confusion matrix' if normalize else 'Confusion matrix, without normalization'

    if normalize:
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        print("Normalized confusion matrix")
        acc_list = [round(cm[i][i], 2) if cm.sum(axis=1)[i] > 0 else None for i in range(cm.shape[0])]
        print("the accuracy of every classes:{}".format(acc_list))
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # 修改处 4：自动生成 ticks 和标签，确保一致性
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = np.max(cm) / 2. if np.max(cm) != 0 else 1.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(os.path.join(path, title + '.png'))
    plt.close()


def draw_ytrue_ypre(ytrue_dict, ypre_dict, save_dir):
	fig, ax = plt.subplots(len(ytrue_dict),1,figsize=(20,3*len(ytrue_dict)))
	for i in range(len(ytrue_dict)):
		ax[i].plot(ytrue_dict[i], label='ytrue')
		ax[i].plot(ypre_dict[i], label='ypre')
		# 设置纵坐标0，1，2，3，4分别对应的标签
		ax[i].set_yticks([0, 1, 2, 3, 4])
		# 设置纵坐标标签
		ax[i].set_yticklabels(['wake', 'N1', 'N2', 'N3', 'REM'])
		# 将ytrue和ypre不同的地方标记出来
		# for j in range(len(ytrue_dict[i])):
		# 	if ytrue_dict[i][j] == ypre_dict[i][j]:
		# 		ax[i].scatter(j, ytrue_dict[i][j], c='g', marker='o')
		ax[i].set_title(f"sub_{i}")
		ax[i].legend()
	plt.savefig(os.path.join(save_dir, f'ytrue_ypre.png'))   

if __name__ == "__main__":
    args = get_args()
    # 构建结果文件夹
    folder_name = args.folder_name
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
     # 设置日志
    logger = get_logger(os.path.join(folder_name, args.log_name), name="test_run")
    # 设置随机种子
    set_seed(args)
    # 打印参数
    logger.info('*'*25 + ' args ' + '*'*25)
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')
    
    preprocesser = DataPreprocesser(args.data_path, args.channels, args.dataset, args.input_epoch_num, 
                                    args.sampling_rate, args.label_available,args)
    test_dataset = preprocesser.gen_testdata()
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # 打印数据集大小
    logger.info(f"test dataset size: {len(test_dataset)}")
    # 打印数据集形状
    logger.info(f"test dataset shape: {test_dataset[0][0].shape}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # net = torch.load("model.pth").to(device)
    net = TinySalientSleepNet(5, [16, 32, 32, 64, 128], [10, 5, 5, 5], 1, 30 * 100, 120, 5, [16, 16, 32, 64, 128]).to(device)
    net=torch.load(os.path.join(args.model_path, "fold0_model.pkl"),map_location=device).to(device)
    # net=torch.load(os.path.join(args.model_path, "model.pth"),map_location=device).to(device)
    # net.load_state_dict(torch.load(os.path.join(args.model_path, "model.pth"),map_location=device))
    ypred, ytrue = test_result(net, test_loader, preprocesser.window_num_by_sub, preprocesser.last_epoch_num_by_sub, device, args)
    # 将list yred按照每列一个子被试的形式保存，子被试的长度不一样
    # 找到最长的被试预测长度
    # max_len = max(len(arr) for arr in ypred)
    # 把所有子被试的预测结果补到一样长
    # padded_preds = []
    # for arr in ypred:
    #     # 补NaN到最长
    #     padded = np.pad(arr, (0, max_len - len(arr)), constant_values=-1)
    #     padded_preds.append(padded)
    # 转成DataFrame，每列一个子被试
    # ypred_df = pd.DataFrame(np.stack(padded_preds, axis=1))  # 注意是axis=1，这样列是被试

    # 保存
    # ypred_df.to_csv(os.path.join(folder_name, "ypred.csv"), index=False)
    p_names = [i for i in os.listdir(args.data_path) if i!="7E3A1B9A-E84C-4B34-A1E4-272FEAED0FA8" and i!="9F6FCBA2-978E-4F6F-937D-6072AECBB5A4"]
    p_names.sort()
    labels = ['W', 'N1', 'N2', 'N3', 'REM']
    
    # 每个epoch的时间长度，30秒 = 30000毫秒
    epoch_duration_ms = 30000
    # 保存每个子被试的预测标签
    for sub,pre in enumerate(ypred):
        # 初始化
        result = []
        for i,p in enumerate(pre):
            # ypred_df[i].to_csv(os.path.join(folder_name, f"ypred_{p_names[i]}.csv"), index=False)
            # 构建json
            entry = {
                "label_name": labels[int(p)],
                "begin_time": i * epoch_duration_ms,
                "end_time": (i + 1) * epoch_duration_ms,
                "epoch_index": i + 1  # 从1开始
            }
            result.append(entry) 
        # with open(os.path.join(folder_name,f'label_{p_names[sub]}.json'), 'w') as f:
        #     json.dump(result, f, indent=4)

    # 计算指标
    if args.label_available:
        # 保存ytrue和ypred
        data={
            "ypred": ypred,
            "ytrue": ytrue
        }
        with open(os.path.join(folder_name, "ytrue_ypred.pkl"), 'wb') as f:
            pkl.dump(data, f)
        # 计算每个子被试的准确率
        acc_, f1_,wf1_, precision_, recall_ = [], [], [], [], []
        # f1_allclass_ = []
        label_all = ['wake', 'N1', 'N2', 'N3', 'REM']
        for i in range(len(preprocesser.window_num_by_sub)):
            # label_sub = [label_all[int(i)] for i in np.unique(ytrue[i])]
            cm, _ = plot_and_print_cm(ypred[i], ytrue[i], folder_name, label_all, file_name=f'sub_{i}_')
            acc = accuracy_score(ytrue[i], ypred[i])
            f1 = f1_score(ytrue[i], ypred[i], average='macro')
            wf1 = f1_score(ytrue[i], ypred[i], average='weighted')
            # f1_allclass_.append(f1_score(ytrue[i], ypred[i], average=None))
            precision = precision_score(ytrue[i], ypred[i], average='macro')
            recall = recall_score(ytrue[i], ypred[i], average='macro')
            logger.info(f"sub {i} acc: {acc:5.8f}, f1: {f1:5.8f}, precision: {precision:5.8f}, recall: {recall:5.8f}")
            acc_.append(acc)
            f1_.append(f1)
            wf1_.append(wf1)
            precision_.append(precision)
            recall_.append(recall)
        # logger.info(f"avg acc: {acc_/len(preprocesser.window_num_by_sub):5.8f}, f1: {f1_/len(preprocesser.window_num_by_sub):5.8f}, precision: {precision_/len(preprocesser.window_num_by_sub):5.8f}, recall: {recall_/len(preprocesser.window_num_by_sub):5.8f}")
        logger.info(f"avg acc: {np.mean(acc_):5.8f}, f1: {np.mean(f1_):5.8f}, wf1: {np.mean(wf1_):5.8f}, precision: {np.mean(precision_):5.8f}, recall: {np.mean(recall_):5.8f}")
        # logger.info(f"f1_allclass: {np.mean(f1_allclass_, axis=0)}")
        # 计算整体指标

        ypred_all = np.concatenate(ypred, axis=0)
        ytrue_all = np.concatenate(ytrue, axis=0)
        # 绘制总的混淆矩阵
        cm, _ = plot_and_print_cm(ypred_all, ytrue_all, folder_name, label_all, file_name='all_')
        acc = accuracy_score(ytrue_all, ypred_all)
        f1 = f1_score(ytrue_all, ypred_all, average='macro')
        wf1 = f1_score(ytrue_all, ypred_all, average='weighted')
        precision = precision_score(ytrue_all, ypred_all, average='macro')
        recall = recall_score(ytrue_all, ypred_all, average='macro')
        f1_allclass_all = f1_score(ytrue_all, ypred_all, average=None)
        logger.info(f"all acc: {acc:5.8f}, f1: {f1:5.8f}, wf1: {wf1:5.8f}, precision: {precision:5.8f}, recall: {recall:5.8f}")
        logger.info(f"f1_allclass_all: {f1_allclass_all}")
        # logger.info(f"first 10 avg acc: {np.mean(acc_[:10]):5.8f}, f1: {np.mean(f1_[:10]):5.8f}, precision: {np.mean(precision_[:10]):5.8f}, recall: {np.mean(recall_[:10]):5.8f}")
        # # 计算前10个子被试的指标
        # ypred_all = np.concatenate(ypred[:10], axis=0)
        # ytrue_all = np.concatenate(ytrue[:10], axis=0)
        # acc = accuracy_score(ytrue_all, ypred_all)
        # f1 = f1_score(ytrue_all, ypred_all, average='macro')
        # precision = precision_score(ytrue_all, ypred_all, average='macro')
        # recall = recall_score(ytrue_all, ypred_all, average='macro')
        # logger.info(f"first 10 all acc: {acc:5.8f}, f1: {f1:5.8f}, precision: {precision:5.8f}, recall: {recall:5.8f}")
        
        # 画图
        draw_ytrue_ypre(ytrue, ypred, folder_name)
        
