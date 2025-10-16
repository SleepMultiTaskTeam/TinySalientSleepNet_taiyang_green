import numpy as np
from xml.etree import ElementTree as ET
import pyedflib
import mne


stage_label = {
	"Wake|0": 0,
	"Stage 1 sleep|1": 1,
	"Stage 2 sleep|2": 2,
	"Stage 3 sleep|3": 3,
	"Stage 4 sleep|4": 3,
	"REM sleep|5": 4,
}

def load_shhs_edf(data_dir):
	f = pyedflib.EdfReader("D:\\dataset\\SHHS\\SHHS1\\shhs1-200001.edf")
	n = f.signals_in_file
	print("signal numbers:", n)
	signal_labels = f.getSignalLabels()
	print("Labels:", signal_labels)
	signal_headers = f.getSignalHeaders()
	print("Headers:", signal_headers)

	# 通过C4M1和F4M1这两个导联显示
	raw_data = mne.io.read_raw_edf(data_dir)
	# raw_data.pick_channels(["EEG(sec)", 'EOG(L)', 'EMG'])  # "EOG(L)", "EOG(R)"都可用
	# raw_data.pick_channels(["ECG", 'THOR RES', 'ABDO RES'])
	raw_data.pick_channels(["ECG", 'EEG', 'ABDO RES'])
	# data_info = raw_data.info
	# print(data_info)
	raw_dataframe = raw_data.to_data_frame()

	return raw_dataframe

def load_shhs_xml_stage_label(data_dir):
	tree = ET.parse(data_dir)
	root = tree.getroot()
	for child in root:
		print('Tag:', child.tag)
		print('Text:', child.text)
		print('Attributes:', child.attrib)
	ScoredEvents = root.find('ScoredEvents')

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

	return sleep_stage
