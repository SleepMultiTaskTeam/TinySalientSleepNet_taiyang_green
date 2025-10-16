import pyedflib
import mne
import numpy as np
import xml.dom.minidom
from xml.etree import ElementTree as ET
import neurokit2 as nk
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
	"freq": 125,
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
raw_data = mne.io.read_raw_edf("D:\\dataset\\SHHS\\SHHS1\\shhs1-200001.edf")
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
freq = config["freq"]

xall = raw_dataframe.values[:]
xall = xall.swapaxes(0, 1)
timeline = xall[0]
xall = xall[1:]
# 处理时间较长，可暂时禁用
peaks, info = nk.ecg_peaks(xall[0], sampling_rate=125)
hrv = nk.hrv_time(peaks, sampling_rate=125)

beginCut = 1250
EndCut = 1250 * 4
fig, ax = plt.subplots(nrows=3, ncols=1)
ax[0].plot(timeline[beginCut: EndCut], xall[0][beginCut: EndCut])
ax[0].set_title("ecg")
ax[1].plot(timeline[beginCut: EndCut], xall[1][beginCut: EndCut])
ax[1].set_title("thor")
ax[2].plot(timeline[beginCut: EndCut], xall[2][beginCut: EndCut])
ax[2].set_title("abdo")
plt.show()

while p < len(sleep_stage):
	if p + input_epoch_num < len(sleep_stage):
		x = raw_dataframe.values[p * 30 * freq: (p + input_epoch_num) * 30 * freq]
		y = sleep_stage[p: p + input_epoch_num]

		x = x.swapaxes(0, 1)
		x = x[1:]
		rri, _ = nk.ecg_peaks(x[0], sampling_rate=125)
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

print("a")

