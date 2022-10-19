__author__ = 'YaelSegal & May Arama'


import torch
import argparse
import numpy as np
import os
import glob
import math
from dataset import PredictDataset
import utils
import train_util
from model import load_model


parser = argparse.ArgumentParser(description='predict pitch')



parser.add_argument('--data', type=str, default='/Users/yaels/Downloads/PiMOD/f2nw0000.wav',
					help='data path, can be directory with wav files or single wav file')

parser.add_argument('--model', type=str, default='./model/6_decoders_MDB_KEELE.pth',help='the model to load')

parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',	help='batch size')
parser.add_argument('--seed', type=int, default=1245,	help='random seed')
parser.add_argument('--window_stride', type=float, default=.01, metavar='N',	help='window step')
parser.add_argument('--outpath', type=str, default='OUTPUT_DIRECTORY', metavar='N',	help='outpath')
args = parser.parse_args()

logger = None
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args.cuda = args.cuda and torch.cuda.is_available()
if args.cuda:
	torch.cuda.manual_seed(args.seed)
	device = 'cuda'
else:
	device = 'cpu'

if not os.path.exists(args.outpath):
    os.makedirs(args.outpath)
#  ################################## load model ########################################

best_pitch_model, num_classes, min_pitch, max_pitch, models_parts  = load_model(args.model)


if args.cuda:
	print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))
	best_pitch_model = torch.nn.DataParallel(best_pitch_model).cuda()


#  ################################## load data ########################################

start = 1200 * math.log2(min_pitch/10)
end = 1200 * math.log2(max_pitch/10)
slices_list = utils.get_slices_cents(num_classes, start, end)
if os.path.isdir(args.data):
    files_list = glob.glob(args.data + "/*.wav")
    files_list += glob.glob(args.data + "/*.WAV")
else:
    files_list = [args.data]

for filename in files_list:
    test_dataset = PredictDataset(filename, args.window_stride)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=args.cuda)

    predict_list, conf = train_util.test_predict(test_loader, best_pitch_model, device, models_parts, slices_list, VOICING_THRESHOLD=min_pitch + 1)
    new_filename = os.path.basename(filename).replace(".wav", ".f0")
    file_writer = open(os.path.join(args.outpath, new_filename), "w")
    file_writer.write("time,frequency,confidence\n")
    for time, pitch, conf in zip(test_dataset.file_times, predict_list, conf):
        file_writer.write("{:.3f}, {:.3f}, {:.3f}\n".format(time, pitch, conf))
    file_writer.close()