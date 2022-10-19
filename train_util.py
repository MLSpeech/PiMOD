__author__ = 'YaelSegal & May Arama'
import torch
import numpy as np
from scipy.spatial.distance import pdist, cosine, cdist
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
import mir_eval
import torch.nn.functional as F


def train(train_loader, model, optimizer, criterion, epoch, device, log_interval, print_progress=True ):
	model.train()
	global_epoch_loss = 0
	print_loss = []
	for batch_idx, (raw, label, lens_list, pitch_value) in enumerate(train_loader):
		
		optimizer.zero_grad()
		raw, label = raw.to(device), label.to(device).squeeze()

		if raw.size(0)==1 and len(label.shape)<3:
			label = label.unsqueeze(0)
		if raw.size(1)==1 and len(label.shape)<3:
			label = label.unsqueeze(1)
		enc_vec, output = model(raw)
		loss = criterion(output, label, lens_list)

		loss.backward()

		optimizer.step()
		global_epoch_loss += loss.item()
		print_loss.append(loss.item())
		if print_progress:
			if batch_idx % log_interval == 0:

				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
						epoch, batch_idx * raw.shape[0], len(train_loader.dataset), 100.
						* batch_idx / len(train_loader), np.mean(print_loss)))
				print_loss = []

	return global_epoch_loss / len(train_loader.dataset)

def test(loader, model, criterion, device, slices_list, model_part, p, VOICING_THRESHOLD=33):
	with torch.no_grad():
		model.eval()
		test_loss = 0
		target_list = []
		pred_list = []
		target_real_list = []
		epsilon = 1e-20
		count_right_model = [0,0]
		for batch_idx, (raw, label, lens_list, pitch_value) in enumerate(loader):
			raw, label = raw.to(device), label.to(device)
			if raw.size(0)==1 and len(label.shape)<3:
				label = label.unsqueeze(0)
			if raw.size(1)==1 and len(label.shape)<3:
				label = label.unsqueeze(1)

			enc_vec, output = model(raw)

			loss = criterion(output, label, lens_list)
			test_loss +=  loss
			for pred, target, target_real in zip(output.cpu().numpy(), label.cpu().numpy(), pitch_value.cpu().numpy()):
	
				model_idx = np.argmax(pred[:,0])
				count_right_model[0] += float(np.argmax(pred[:,0]) == np.argmax(target[:,0]))

				count_right_model[1] += 1

				if target_real < 0:
					continue

				if model_idx == 0:
					hz_pred = 0
					hz_target = target_real

				else:
					model_len = lens_list[model_idx][0]
					bins = pred[model_idx][1:model_len]
					model_start_bin, model_end_bin = model_part[model_idx-1] 

					best_idx = int(np.argmax(bins))
					start = max(0, best_idx - 4)
					end = min(model_len-1, best_idx + 5)
					cut_slices_list = slices_list[model_start_bin + start:model_start_bin + end]
					cut_pred = bins[start: end]
					pred_val = np.sum(cut_slices_list * cut_pred) / (np.sum(cut_pred) + epsilon) 
					hz_pred = 10 * np.power(2, pred_val.item()/1200)		
					hz_target = target_real				

				pred_list.append(hz_pred)
				target_list.append(hz_target)
				target_real_list.append(target_real)


		test_loss /= len(loader)

	target_array = np.array(target_list)
	pred_array =  np.array(pred_list)
	time_span = mir_eval.melody.constant_hop_timebase(0.02, len(target_list))
	
	target_v, target_c, pred_v, pred_c = mir_eval.melody.to_cent_voicing(time_span, target_array, time_span,pred_array)
	raw_pitch = mir_eval.melody.raw_pitch_accuracy(target_v, target_c, pred_v, pred_c)
	raw_chroma = mir_eval.melody.raw_chroma_accuracy(target_v, target_c, pred_v, pred_c)

	
	nonzero_freqs = np.logical_and(target_array!=0, pred_v!=0)
	freq_diff_hz = np.abs(target_array - pred_array)[nonzero_freqs]
	gpe = utils.grossPitchError(pred_array, target_array, (np.array(target_real_list) > 0), p)
	precision, recall, histogram, F1, MSE = utils.Accuracy(pred_array, target_array, (np.array(target_real_list) > 0), VOICING_THRESHOLD)

	print('\nTest set: Average loss: {:.4f}, \n '.format(test_loss))
	print('Test set: raw_pitch: {:.4f}, raw_chroma: {:.4f},\n '.format(raw_pitch, raw_chroma))
	print('mean hz diff for voice: {} \n'.format(np.mean(freq_diff_hz)))
	print('MSE: {} \n'.format(MSE))
	print('Gross Pitch Error (voiced frames: {}): {}% \n'.format(sum(i > 0 for i in target_real_list), gpe*100))
	print('Accuracy:\n')
	print('precision: {},  recall: {}, F1: {}\n '.format(precision, recall, F1))
	print('histogram:{}\n '.format(histogram))

	return test_loss.item(), raw_pitch, raw_chroma, gpe, 1 - histogram[1]

def test_predict(loader, model, device, model_part, slices_list,  VOICING_THRESHOLD=33):
	with torch.no_grad():
		model.eval()
		pred_list = []
		conf = []
		epsilon = 1e-20
		lens_list = [x[1]-x[0] +1 for x in model_part] 
		lens_list = [1] + lens_list
		for batch_idx, (raw) in enumerate(loader):
			raw = raw.to(device)
			enc_vec, output = model(raw)
			for pred in output.cpu().numpy():
	
				model_idx = np.argmax(pred[:,0])

				if model_idx == 0:
					hz_pred = 0

				else:
					model_len = lens_list[model_idx]
					bins = pred[model_idx][1:model_len]
					model_start_bin, model_end_bin = model_part[model_idx-1]

					best_idx = int(np.argmax(bins))
					start = max(0, best_idx - 4)
					end = min(model_len-1, best_idx + 5)
					cut_slices_list = slices_list[model_start_bin + start:model_start_bin + end]
					cut_pred = bins[start: end]
					pred_val = np.sum(cut_slices_list * cut_pred) / (np.sum(cut_pred) + epsilon) 

					hz_pred = 10 * np.power(2, pred_val.item()/1200)			

				conf.append(pred[model_idx,0])
				pred_list.append(hz_pred)

	return pred_list, conf

