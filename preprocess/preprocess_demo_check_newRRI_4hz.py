import pyedflib
import mne
import numpy as np
import xml.dom.minidom
from xml.etree import ElementTree as ET
import neurokit2 as nk
from scipy import interpolate
import matplotlib.pyplot as plt

stage_label = {
	"Wake|0": 0,
	"Stage 1 sleep|1": 1,
	"Stage 2 sleep|2": 2,
	"Stage 3 sleep|3": 3,
	"Stage 4 sleep|4": 3,
	"REM sleep|5": 4,
}

config = {
	"input_epoch_num": 10,
	"sampling_rate": 125,
}


tree = ET.parse("D:\\dataset\\SHHS\\SHHS1\\shhs1-200001-nsrr.xml")
root = tree.getroot()
for child in root:
	print('Tag:', child.tag)
	print('Text:', child.text)
	print('Attributes:', child.attrib)
ScoredEvents = root.find('ScoredEvents')

beginFlag = True
endTime = 0.0
sleep_stage = []
for ScoredEvent in ScoredEvents:
	event = ScoredEvent.find('EventType')
	if event.text == 'Stages|Stages':
		assert float(ScoredEvent.find('Start').text) == endTime
		duration = float(ScoredEvent.find('Duration').text)
		label = stage_label[ScoredEvent.find('EventConcept').text]
		# 单位是秒
		endTime = duration + endTime
		epoch_num = int(duration // 30.0)
		for i in range(epoch_num):
			sleep_stage.append(label)

sleep_stage = np.array(sleep_stage)


f = pyedflib.EdfReader("D:\\dataset\\SHHS\\SHHS1\\shhs1-200001.edf")
n = f.signals_in_file
print("signal numbers:", n)
signal_labels = f.getSignalLabels()
print("Labels:", signal_labels)
signal_headers = f.getSignalHeaders()
print("Headers:", signal_headers)

#通过C4M1和F4M1这两个导联显示
raw_data = mne.io.read_raw_edf("D:\\dataset\\SHHS\\SHHS1\\shhs1-200005.edf")
raw_data.pick_channels(["ECG", 'THOR RES', 'ABDO RES'])  # "EOG(L)", "EOG(R)"都可用
data_info = raw_data.info
print(data_info)
raw_dataframe = raw_data.to_data_frame()
# raw_dataframe = raw_dataframe.values[:,:]

input_time_window = 30.0 * config["input_epoch_num"]
timeReminder = input_time_window
x_group = None
y_group = None
xpos = 0

p = 0
input_epoch_num = config["input_epoch_num"]
sampling_rate = config["sampling_rate"]

xall = raw_dataframe.values[:]
xall = xall.swapaxes(0, 1)
time_list = xall[0]
xall = xall[1:]
raw_ecg = xall[0]
raw_thor = xall[1]
raw_abdo = xall[2]
ecg_cleaned = nk.ecg_clean(raw_ecg, sampling_rate=sampling_rate)
# plt.plot(xall[0][-1002000:-1000000])

# plt.plot(ecg_cleaned[1000000:1002000])
# plt.plot(ecg_cleaned[-1002000:-1000000]) # 1000000 / 125 = 8000s, 8000s / 3600 = 2.22h

# plt.plot(ecg_cleaned[1000000:1002000])
# plt.plot(ecg_cleaned[-2000:])

# fig,ax = plt.subplots(nrows=3, ncols=1)
# ax[0].plot(ecg_cleaned[1000000:1002000])
# ax[0].set_title("ecg_cleaned[1000000:1002000]")
#
# ax[1].plot(ecg_cleaned[-1002000:-1000000])
# ax[1].set_title("ecg_cleaned[-1002000:-1000000]")
#
# ax[2].plot(ecg_cleaned[-2000:])
# ax[2].set_title("ecg_cleaned[-2000:]")
# plt.show()

_, info = nk.ecg_peaks(xall[0], sampling_rate=sampling_rate)
peaks = info["ECG_R_Peaks"]

# Convert peaks to intervals
rri = np.diff(peaks) / sampling_rate * 1000  # 毫秒单位的RRI
rri_time = peaks[1:] # 以采样点为单位的时间轴，RRI会舍弃第一个峰，从第二个峰开始计算
rri_real_time = np.array(peaks[1:]) / sampling_rate
# plt.plot(rri_real_time, rri)
# plt.show()

fig,ax = plt.subplots(nrows=2, ncols=1)
ax[0].plot(rri_time, rri)
ax[0].set_title("RRI")
ax[1].plot(ecg_cleaned)
ax[1].set_title("ECG")
plt.show()

# 插值
f = interpolate.interp1d(rri_real_time, rri, kind="linear")  # 生成插值模板
new_rri_real_time = np.arange(0, rri_real_time[-1], 0.25)  # 生成插值的时间点
# 低于rri_real_time[0]的点要舍弃
new_rri_real_time = new_rri_real_time[new_rri_real_time >= rri_real_time[0]]
new_rri = f(new_rri_real_time)

fig,ax = plt.subplots(nrows=2, ncols=1)
begin,end = 0,100
ax[0].plot(new_rri_real_time[begin:end], new_rri[begin:end])
# ax[0].set_title("RRI ["+begin+":"+end+"]")
#  np.where(time_list <= new_rri_real_time[end])
# ax[1].plot(time_list, raw_ecg[begin*125:end*125])
ax[1].plot(ecg_cleaned[begin*125:end*125])
# ax[1].set_title("ECG ["+begin+":"+end+"]")
plt.show()

# 由于RRI和插值的错位，第一个睡眠期和最后一个睡眠期整个裁剪掉
# 剔除前后30min，前和后因为之前提取RRI都多剃掉了一些，所以这些取消掉来对齐。
# 已核验，剔除正确
left_cut = 30 * 60 * sampling_rate - rri_time[0]
right_cut = - (30 * 60 * sampling_rate - (len(raw_ecg) - rri_time[-1] - 1))
new_rri_time = new_rri_time[left_cut:right_cut]
new_rri = new_rri[left_cut:right_cut]
new_rri_real_time = new_rri_real_time[left_cut:right_cut]

# 睡眠期剔除前后30min，前和后分别是30*60/30 = 60个epoch
sleep_stage = sleep_stage[60:-60]

# 另外两个信号也剔除前后30min
left_cut = 30 * 60 * sampling_rate
right_cut = - (30 * 60 * sampling_rate)
new_thor = raw_thor[left_cut:right_cut]
new_abdo = raw_abdo[left_cut:right_cut]

# 结合，如果之前RRI剔除前后30min的计算有误，这里是不能合起来的，尺寸不对
new_xall = np.concatenate([new_rri[np.newaxis, :], new_thor[np.newaxis, :], new_abdo[np.newaxis, :]], axis=0)
new_xall = new_xall.swapaxes(0, 1)  # 时间轴,3通道

# 切分x和睡眠期
while p < len(sleep_stage):
	if p + input_epoch_num < len(sleep_stage):
		x = new_xall[p * 30 * sampling_rate: (p + input_epoch_num) * 30 * sampling_rate]
		y = sleep_stage[p: p + input_epoch_num]

		x = x.swapaxes(0, 1)

	else:
		break  # 多出的一段不要了
	p += input_epoch_num

	if x_group is None:
		x_group = x[np.newaxis, :]
		y_group = y[np.newaxis, :]
	else:
		x_group = np.append(x_group, x[np.newaxis, :], axis=0)
		y_group = np.append(y_group, y[np.newaxis, :], axis=0)

print("a")

