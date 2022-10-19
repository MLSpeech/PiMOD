import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import LambdaLayer, padd_list_tensors


class PitchPred(nn.Module):
	def __init__(self, input_size=512, hidden_layer=256, bins=100, dropout=0):
		super(PitchPred, self).__init__()
		self.input_size = input_size
		self.fc1 = nn.Linear(input_size, hidden_layer)
		self.dropout = nn.Dropout(dropout)
		self.bins = bins
		self.fc2 = nn.Linear(hidden_layer, bins+1)
		self.sig = nn.Sigmoid()


	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = self.dropout(x)
		probs = self.sig(self.fc2(x))
		return probs

class CNNEncoder(nn.Module):
	def __init__(self, ninput=512, channels=256, dropout=0.25):
		super(CNNEncoder, self).__init__()
		self.enc = nn.Sequential(
			nn.Conv1d(1, channels, kernel_size=512, stride=4, padding=0, bias=False),
			nn.BatchNorm1d(channels),
			nn.LeakyReLU(),
			nn.Dropout(p=dropout),
			nn.Conv1d(channels, 128, kernel_size=64, stride=2, padding=31, bias=False),
			nn.BatchNorm1d(128),
			nn.LeakyReLU(),
			nn.Dropout(p=dropout),
			nn.Conv1d(128, 128, kernel_size=32, stride=2, padding=0, bias=False),
			nn.BatchNorm1d(128),
			nn.LeakyReLU(),
			nn.Dropout(p=dropout),
			nn.Conv1d(128, 128, kernel_size=16, stride=2, padding=7, bias=False),
			nn.BatchNorm1d(128),
			nn.LeakyReLU(),
			nn.Dropout(p=dropout),
			nn.Conv1d(128, ninput, kernel_size=8, stride=2, padding=3, bias=False),
			nn.BatchNorm1d(ninput),
			nn.LeakyReLU(),
			nn.MaxPool1d(kernel_size=4),
			LambdaLayer(lambda x: x.transpose(1,2)),
		)
		self.ninput = ninput

	def forward(self, raw_input):
		x = raw_input
		for m in self.enc.children():
			x = m(x)

		enc_input = self.enc(raw_input)
		return enc_input.squeeze()

class PiMOD(nn.Module):

	def __init__(self, ninput=512, channels=256, dropout=0.25, hidden_layer=256, parts=[]):
		super(PiMOD, self).__init__()
		self.ninput = ninput
		self.channels = channels
		self.hidden_layer = hidden_layer
		self.enc = CNNEncoder(ninput=ninput, channels=channels, dropout=dropout)
		self.linears = nn.ModuleList()
		self.parts =[]
		# add pred model for silence
		self.linears.append(PitchPred(input_size=ninput, bins=0))
		self.parts.append(1)
		for class_item in parts:

			self.linears.append(PitchPred(input_size=ninput, bins=class_item))
			self.parts.append(class_item + 1)

	def forward(self, x):
		enc_vec = self.enc(x)
		total_vec_list = []
		for i, small_model in enumerate(self.linears):
			cur_model_out = small_model(enc_vec)    
			if len(cur_model_out.shape)<2:
				cur_model_out = cur_model_out.unsqueeze(0)
			total_vec_list.append(cur_model_out) 
		all_models_out_tensor = padd_list_tensors(total_vec_list, self.parts, 1)


		return enc_vec, all_models_out_tensor.transpose(0,1)


def load_model(path):
	checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
	params = checkpoint['params']

	pitch_model = PiMOD(ninput=params["input_size"], channels=params["channels"], dropout=0, hidden_layer=params["hidden_layer"], parts=params["parts_len"] )

	pitch_model.load_state_dict(checkpoint['net'])
	min_pitch = params["min_pitch_value"]
	max_pitch = params["max_pitch_value"]
	parts_len = params["parts_len"]
	models_parts = params["models_parts"]
	num_classes = params["num_classes"]

	return pitch_model, num_classes, min_pitch, max_pitch, models_parts