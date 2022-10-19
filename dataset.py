
import os
import os.path

import librosa
import numpy as np
import torch
import torch.utils.data as data
import soundfile
import utils
import random
import math
import glob
SR = 16000


def limit_raw_loader(wav_path=None, y=None, window_size=.064, window_stride=.01, select_channel=1, pitch_idx=-1, padd = 0):

	if wav_path:
		y, sr = soundfile.read(wav_path)  # much faster than librosa!
		if select_channel > 0:
			y = np.asfortranarray(y[:,select_channel])
		if sr > 16000:
			y = librosa.resample(y, sr, 16000)
			sr = 16000

	if pitch_idx >=0:
		pitch_y_idx = pitch_idx * 16
		padd_y = 16 * padd
		start_idx = max(pitch_y_idx - padd_y, 0)
		padd_zero = min((pitch_y_idx - padd_y), 0)
		end_idx = min(pitch_y_idx + padd_y, len(y))
		y = y[start_idx:end_idx]
		if padd_zero < 0:
			start_idx = padd_zero
			padd_zero = -padd_zero
		real_idx = (pitch_y_idx - start_idx)/16
		
	raw_list = []
	start_list = []
	start = 0
	hop_length = int(SR * window_stride)
	sr_window_size = int(SR * window_size)
	y = np.pad(y, (padd_zero, int(sr_window_size/2)), 'constant', constant_values=(0, 0))
	while start < len(y) - sr_window_size:
		raw_list.append(y[start: start + sr_window_size])
		start_list.append(start)
		start += hop_length

	real_idx = start_list.index(7680)
	raw_array = np.asarray(raw_list)
	raw_array = raw_array.transpose()
	mean =  np.mean(raw_array, axis=0)
	raw_array -= mean
	raw_array = raw_array.transpose()


	raw_tensor = torch.FloatTensor(raw_array)
	return raw_tensor, real_idx

def calc_step(labels_lines, step_size):
	pitches = []
	step_size = step_size/ 1000
	start = float(labels_lines[0].split(",")[0])
	if start > 0:
		tmp_start = 0
		while tmp_start <start:
			pitches.append(0)
			tmp_start += step_size

	diff = float(labels_lines[1].split(",")[0]) - float(labels_lines[0].split(",")[0])
	if diff > step_size:
		raise "step size to small to dataset"
	prev_pitch = 0
	prev_time = 0

	for line in labels_lines:
		time, pitch = line.strip("\n").split(",")
		time ,pitch = float(time), float(pitch)

		if time == start:
			pitches.append(pitch)
			start += step_size
			start = round(start, 6)
		else:
			if time > start:
				if abs(time - start) < abs(prev_time - start):
					pitches.append(pitch)
				else:
					pitches.append(prev_pitch)
				prev_time = time
				prev_pitch = pitch
				start += step_size
				start = round(start, 6)
			else: # time < start
				prev_time = time
				prev_pitch = pitch
	return pitches

def create_pairs(files_list, files2pitches):
	# just use all the pitch values in the dataset
	file_label_pairs = []
	for file_idx, item in enumerate(files_list):
		wav, labels_file = item
		pitches = files2pitches[wav]
		for pitch_idx in range(len(pitches)):
			file_label_pairs.append([file_idx, pitch_idx])

	return file_label_pairs

def make_dataset(data_path, folds, step_size):

	dataset = []
	file_to_singer_dict = {}
	files2pitches = {}
	count = 0
	for fold in os.listdir(data_path):
		# takes only the selected folds
		fold_num = fold.split("_")[-1]
		if not fold_num in folds:
			continue

		wavs_list = glob.glob(os.path.join(data_path, fold) + "/*.wav")

		for wav in wavs_list:
	
			count += 1
			current_label_file = wav.replace(".wav", ".txt")
			wav_filename = wav.split("/")[-1]
			labels_lines = open(current_label_file, "r").readlines()
			pitches = calc_step(labels_lines, step_size)

			files2pitches[wav] = pitches
			dataset.append([wav, current_label_file])
			wav_name = os.path.basename(wav).replace(".wav", "")
			singer = wav_name.split("_")[0]
			file_to_singer_dict[wav] = singer


	return dataset, file_to_singer_dict, files2pitches


class TrainDataset(data.Dataset):
	"""
	"""
	def create_data(self, data_path):


		dataset, file_to_singer_dict, files_pitches_dict = make_dataset(data_path, self.folds, self.window_stride*1000)

		file_label_pairs = create_pairs(dataset, files_pitches_dict)

		return file_label_pairs, dataset, file_to_singer_dict, files_pitches_dict

	def __init__(self, data_path, folds, seed, input_slices, models_parts=[[0,26],[23,56],[53, -1]], window_stride=.02, predict=False, min_hz_value=32.7):
		np.random.seed(seed)
		random.seed(seed)
		self.folds = folds # list of selected folds
		self.window_stride = window_stride
		self.padd = 512
		self.input_slices = input_slices
		self.models_parts = models_parts
		self.predict=predict

		if type(data_path) is list:
			dataset = []
			pairs = []
			file_to_singer_dict = {}
			files_pitches_dict = {}
			for item in data_path:
				item_pairs, item_dataset, item_file_to_singer_dict, item_files_pitches_dict = self.create_data(item)
				dataset.extend(item_dataset)
				pairs.extend(item_pairs)
				file_to_singer_dict.update(item_file_to_singer_dict)
				files_pitches_dict.update(item_files_pitches_dict)
		else:

			pairs, dataset, file_to_singer_dict, files_pitches_dict = self.create_data(data_path)

		self.min_hz_value = min_hz_value
		self.file_label_pairs = pairs
		self.files_pitches_dict = files_pitches_dict


		count = [0 for i in range(len(input_slices)+1)]
		for file_idx, pitch_idx in self.file_label_pairs:
			pitch_val = self.files_pitches_dict[dataset[file_idx][0]][pitch_idx]
			try:
				pitch_val = 1200 * math.log2(pitch_val/10)
				diff_vals = input_slices - pitch_val
				count[np.argmax(diff_vals >= 0)+1] +=1
			except:
				pitch_val = 0
				count[0] +=1



		print("models %")
		print(f"range idx: {count[0]} , examples: {count[0]}, {count[0]/ sum(count)}%" )
		for idx, (start, end) in enumerate(models_parts):
			print(f"range idx: {idx+1}, examples: {sum(count[start+1:end])}, {sum(count[start+1:end])/ sum(count)}%" )
		self.dataset = dataset
		self.file_to_singer_dict = file_to_singer_dict

		print("dataset len: {}".format(len(self.file_label_pairs)))

	def gaussian(self, slice_pitch, pitch_val):

		return math.exp(-(slice_pitch - pitch_val)**2 / (2 *25**2))


	def __getitem__(self, index):
		"""
		Args:
			index (int): Index
		Returns:
			tuple: (spect, target) where target is class_index of the target class.
		"""
		file_idx , pitch_idx = self.file_label_pairs[index]
		wav_path, label_path = self.dataset[file_idx]
		step_size = 1000 * self.window_stride

		raw, real_new_idx = limit_raw_loader(wav_path, y=None, window_stride=self.window_stride, select_channel=0,
											pitch_idx=int(pitch_idx*step_size), padd=self.padd)
		new_pitch_idx = real_new_idx
		pitches = self.files_pitches_dict[wav_path]
		pitch_value = pitches[pitch_idx]
		
		try:
			pitch_calc = 1200 * math.log2(pitch_value/10)
		except:
			pitch_calc = 1200 * math.log2(self.min_hz_value/10)


		target = []
		weights = []
		for slice_pitch in self.input_slices:
			if  pitch_value ==0:
				y_i = 0
			else:
				y_i = self.gaussian(slice_pitch, pitch_calc)
			target.append(y_i)
			if y_i > 1e-3:
				weights.append(1)
			else:
				weights.append(0)

		weight_target_list = []
		lens_list = []

		prec = [1] if pitch_value ==0 else [0]
		weight_target_list.append(torch.FloatTensor(prec))
		lens_list.append(len(prec))

		for start,end in self.models_parts:
			try:
				prec = [sum(weights[start:end])/sum(weights)] # calc the value for the mse
			except:
				prec = [0] # calc the value for the mse
			prec.extend(target[start:end])
			weight_target_list.append(torch.FloatTensor(prec))
			lens_list.append(len(prec))


		pitch_y_tensor = utils.padd_list_tensors(weight_target_list,lens_list, 0)
		if len(pitch_y_tensor.shape)<2:
			pitch_y_tensor = pitch_y_tensor.unsqueeze(0)	

		return raw[new_pitch_idx, :].unsqueeze(0), pitch_y_tensor, lens_list, pitch_value


	def __len__(self):
		return len(self.file_label_pairs)

	def get_label(self, idx):
		return self.file_to_singer_dict[self.dataset[idx][0]]


class PredictDataset(data.Dataset):
	"""
	"""

	def __init__(self, filename, window_stride):

		self.window_stride = window_stride
		self.padd = 512
		self.dataset = []
		self.file_pairs = []
		self.filename = filename

		f = soundfile.SoundFile(filename)
		
		if f.channels > 1:
			raise Exception(f"{filename} has more than one channel")

		y = f.read()
		sr = f.samplerate
		if f.samplerate > SR:
			y = librosa.resample(y, orig_sr=f.samplerate, target_sr=SR)
			print(f"{filename} resample from {f.samplerate} to {SR}")
			sr = SR
		elif f.samplerate < SR:
			raise Exception(f"{filename} sample rate is smaller that {SR}")
		

		self.y = y
		wav_duration = len(y) / sr
		file_times = np.arange(0, wav_duration, window_stride)
		self.file_times = file_times
		self.dataset = np.arange(0, len(file_times), 1)
		print('filename :{}'.format(filename))
		print('sample rate = {}'.format(sr))
		print('seconds = {}'.format(wav_duration))

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index
		Returns:
			tuple: (spect, target) where target is class_index of the target class.
		"""
		pitch_idx = self.dataset[index]
		wav_path = self.filename
		step_size = 1000 * self.window_stride

		raw, real_new_idx = limit_raw_loader( y=self.y, wav_path=None, window_stride=self.window_stride, select_channel=0,
											pitch_idx=int(pitch_idx*step_size), padd=self.padd)

		new_pitch_idx = real_new_idx

		return raw[new_pitch_idx, :].unsqueeze(0)


	def __len__(self):
		return len(self.dataset)

