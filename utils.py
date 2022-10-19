
import torch
import torch.nn as nn
import numpy as np


class LambdaLayer(nn.Module):
    # Felix Kreuk code
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad
    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """

    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)

    return torch.cat([vec, torch.zeros(*pad_size).to(vec.device)], dim=dim)


def padd_list_tensors(targets, targets_lens, dim):

    target_max_len = max(targets_lens)
    padded_tensors_list = []
    for tensor, tensor_len in zip(targets, targets_lens):
        pad = pad_tensor(tensor, target_max_len, dim)
        padded_tensors_list.append(pad)
    padded_tensors = torch.stack(padded_tensors_list)
    return padded_tensors

def get_slices_cents(size,start_hz, max_hz):
    return np.linspace(start_hz, max_hz, size)

def grossPitchError(pred_pitch, target_pitch, target_v, p):
    error_val = 0
    voicing_val = 0
    for pred, target, voicing in zip(pred_pitch, target_pitch, target_v):
        if not voicing:
            continue
        deviation = (target / 100) * p
        voicing_val += 1
        if abs(pred - target) > deviation:
            error_val += 1
    gpe = error_val/voicing_val if voicing_val > 0 else 0
    return gpe

def Accuracy(pred_pitch, target_pitch, target_v, VOICING_THRESHOLD):
    diff_vals = []
    real_vals = []
    voicing_values = [0, 0, 0, 0] # TP, FP, TN, FN
    for idx, (pred, target, voicing) in enumerate(zip(pred_pitch, target_pitch, target_v)):
        if voicing:
            if pred > VOICING_THRESHOLD:
                voicing_values[0] +=1
            else:
                voicing_values[3] +=1 
                continue
        else:
            if pred > VOICING_THRESHOLD:
                voicing_values[1] +=1
                continue
            else:
                voicing_values[2] +=1
                continue
        
        diff_vals.append(abs(pred-target))
        real_vals.append([idx,pred, target])

    precision = voicing_values[0] / (voicing_values[0] + voicing_values[1]) if (voicing_values[0] + voicing_values[1]) >0 else 0
    recall = voicing_values[0] / (voicing_values[0] + voicing_values[3]) if (voicing_values[0] + voicing_values[3]) > 0 else 0
    
    diff_vals = np.array(diff_vals)
    mse = np.mean(np.square(diff_vals)) 
    sorted_vals = np.sort(diff_vals)

    miss_true = (voicing_values[0] + voicing_values[3])
    thresholds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    histogram = [0]
    raw_num = [0]
    if len(sorted_vals) >0:
        for thres in thresholds:
            min_idx = np.argmax( sorted_vals>thres)
            histogram.append(min_idx/miss_true)
            raw_num.append(min_idx)
    else:
        histogram.extend([0,0])

    histogram.append(1)
    raw_num.append(len(sorted_vals))
    histogram = np.array(histogram)
    raw_num = np.diff(np.array(raw_num))
    F1 = ((precision * recall)/(precision + recall)) * 2 if precision + recall != 0 else 0
    return precision, recall, histogram, F1, mse
