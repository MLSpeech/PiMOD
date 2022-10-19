__author__ = 'YaelSegal & May Arama'

import neptune
import torch.optim as optim
import torch
import argparse
import dataset
import numpy as np
import os
import model   
import train_util
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
import math
import random
import loss


parser = argparse.ArgumentParser(description='train PiMOD')

parser.add_argument('--data', type=str, default='',
                    help='dataset directory')
parser.add_argument('--train_folds', type=str, default='1_2_3',
                    help='folds folders to train on')
parser.add_argument('--val_folds', type=str, default='4',
                    help='folds folders to validate on')
parser.add_argument('--test_folds', type=str, default='5',
                    help='folds folders to testing on')
parser.add_argument('--experiment_name', type=str, default='', help='experiment name')
parser.add_argument('--save_dir', type=str, default='./models',
                    help='directory to save the model')
parser.add_argument('--model_name', type=str, default='', help='directory to save the model')
parser.add_argument('--num_classes', type=int, default=360, metavar='N',	help='num_classes')
parser.add_argument('--channels', type=int, default=512, metavar='N',	help='channels')
parser.add_argument('--hidden_layer', type=int, default=256, metavar='N',	help='hidden_layer')
parser.add_argument('--input_size', type=int, default=512, metavar='N',	help='input_size')
parser.add_argument('--dropout', type=float, default=0.25, metavar='N',	help='dropout')
parser.add_argument('--opt', type=str, default='adam',	help='optimization method: adam || sgd')
parser.add_argument('--lamda', type=float, default=0.2, metavar='N',	help='lamda')
parser.add_argument('--momentum', type=float, default='0.9',	help='momentum')
parser.add_argument('--lr', type=float, default= 0.0002,	help='initial learning rate')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--epochs', type=int, default=500,	help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32,	help='batch size')
parser.add_argument('--seed', type=int, default=1245,	help='random seed')
parser.add_argument('--patience', type=int, default=32,	help='how many epochs of no loss improvement should we wait before stop training')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',	help='report interval')
parser.add_argument('--measure', type=str, default="pitch", metavar='N',	help='pitch || loss || hist')
parser.add_argument('--window_stride', type=float, default=.01, metavar='N',	help='window step')
parser.add_argument('--max_pitch_value', type=float, default=1975.5, metavar='N',	help='max pitch value')
parser.add_argument('--min_pitch_value', type=float, default=32.7, metavar='N',	help='max pitch value')
parser.add_argument('--decoders_weights',type=str, default='1_2_1_1_1_1_2', metavar='N',	help='max pitch value')
parser.add_argument('--p', type=int, default=10,	help='the deviation of estimated pitch frequency from the ground truth is more than p%')
parser.add_argument('--neptune', action='store_true', help='log to neptune')
parser.add_argument('--print_range', action='store_true', help='print decoders pitch range')


args = parser.parse_args()


models_parts = [[0, 56], [54, 115], [113, 174], [172, 233], [231, 292], [290, 360]] 


def build_model_name(args):
    args_dict = [ "data", "lr", "min_pitch_value","max_pitch_value", "measure", "train_folds", "lamda"]
    full_name = "{}_models_with".format(len(models_parts))
    for arg in args_dict:
        full_name += str(arg) + "_" + str(getattr(args, arg)) + "_"

    return full_name + ".pth"

if args.neptune:
    
    neptune.init('user/experiment-name', api_token='TOKEN')
    args_dict = vars(args).copy()
    args_dict['model_parts'] = str(models_parts)
    if not args.model_name:
        description = build_model_name(args)
    else:
        description = ""

    neptune.create_experiment(name=args.experiment_name, params=args_dict, description=description)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# [[0, 13]]
args.cuda = args.cuda and torch.cuda.is_available()
if args.cuda:
    device = 'cuda'
else:
    device = 'cpu'


start = 1200 * math.log2(args.min_pitch_value/10)
end = 1200 * math.log2(args.max_pitch_value/10)
slices_list = utils.get_slices_cents(args.num_classes, start, end)
#  ################################## load model ########################################

parts = [len(slices_list[x[0]:x[1]]) for x in models_parts]
if args.print_range:
    # print decoders range
    
    for start, end in models_parts:

        start = 10 * np.power(2, slices_list[start]/1200)
        end = 10 * np.power(2, slices_list[end-1]/1200)
        print("model range: {}, {}".format(start,end))
    

pitch_model = model.PiMOD( ninput=args.input_size, channels=args.channels, dropout=args.dropout, hidden_layer=args.hidden_layer, parts= parts)


if args.cuda:
    print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))
    pitch_model = torch.nn.DataParallel(pitch_model).cuda()

#  ################################## define optimizer ########################################

if args.opt.lower() == 'adam':
    optimizer = optim.Adam(pitch_model.parameters(), lr=args.lr)
elif args.opt.lower() == 'sgd':
    optimizer = optim.SGD(pitch_model.parameters(), lr=args.lr,  momentum=args.momentum)
else:
    optimizer = optim.SGD(pitch_model.parameters(), lr=args.lr,  momentum=args.momentum)

#  ################################## load data ########################################


train_folds_list = args.train_folds.split("_")

valid_folds_list = args.val_folds.split("_")

train_dataset = dataset.TrainDataset(args.data, train_folds_list, args.seed, slices_list, 
            models_parts=models_parts, window_stride= args.window_stride, min_hz_value=args.min_pitch_value)

val_dataset = dataset.TrainDataset(args.data, valid_folds_list , args.seed, slices_list, 
            models_parts=models_parts, window_stride= args.window_stride,min_hz_value=args.min_pitch_value)


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=20, pin_memory=args.cuda)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=20, pin_memory=args.cuda)

#  ################################## create loss ########################################

if args.decoders_weights:
    weights = [float(w) for w in args.decoders_weights.split("_")]
    criterion = loss.MultBinaryLoss(lamda=args.lamda, decoders_weights=weights)
else:
    criterion = loss.MultBinaryLoss(lamda=args.lamda)

#  ################################## run epochs ########################################
epoch = 1
iteration = 0
# best_loss = np.inf
best_measure = np.inf
total_val_loss, total_raw_pitch, total_row_chrome, total_gpe = [], [], [], []
while (epoch < args.epochs + 1) and (iteration < args.patience):

    
    train_util.train(train_loader, pitch_model, optimizer, criterion, epoch ,device, args.log_interval)
    current_loss, raw_pitch, row_chrome, gpe, hist_5 = train_util.test(val_loader, pitch_model, criterion,
                                                    device, slices_list, models_parts, args.p,VOICING_THRESHOLD=args.min_pitch_value +1)


    total_val_loss.append(current_loss)
    total_raw_pitch.append(raw_pitch)
    total_row_chrome.append(row_chrome)
    total_gpe.append(gpe)


    if args.measure == "hist":
        current_measure = hist_5
    elif args.measure == "pitch":
        current_measure = 1 - raw_pitch
    elif args.measure == "gpe":
        current_measure = gpe
    else:
         current_measure = current_loss
    
    if args.neptune:
        neptune.log_metric('val_raw_pitch', raw_pitch)
        neptune.log_metric('val_row_chrome', row_chrome)
        neptune.log_metric('val_loss', current_loss)
        neptune.log_metric('val_gpe', gpe)
        neptune.log_metric('val_hist_5', hist_5)
    

    if current_measure > best_measure:
        iteration += 1

        print('measure was not improved, iteration {0}'.format(str(iteration)))
    else:
        best_measure = current_measure
        iteration = 0 #

        print('Saving model...         loss has improved, putting patience back to zero')

        state = {
            'net': pitch_model.module.state_dict() if args.cuda else pitch_model.state_dict(), #correct if using "dataparallel"
            'acc': current_loss,
            'epoch': epoch,
            'params': {  "input_size":args.input_size, "max_pitch_value": args.max_pitch_value, "min_pitch_value": args.min_pitch_value, 
            "channels": args.channels, "hidden_layer":args.hidden_layer, "num_classes": args.num_classes, "parts_len":parts, "models_parts":models_parts} 
            }
        if not os.path.isdir(args.save_dir):
            os.mkdir(args.save_dir)

        if args.model_name:
            torch.save(state,  args.save_dir +'/' + args.model_name)
        else:
            torch.save(state,  args.save_dir +'/' + build_model_name(args))
    epoch += 1



############################## testing #########################################

if args.model_name:
    best_model_path =  args.save_dir +'/' + args.model_name
else:
    best_model_path =  args.save_dir +'/' + build_model_name(args)

best_pitch_model, num_classes, min_pitch, max_pitch, models_parts = model.load_model(best_model_path) 
if args.cuda:
    best_pitch_model = torch.nn.DataParallel(best_pitch_model).cuda()

test_folds_list = args.test_folds.split("_")


test_dataset = dataset.TrainDataset(args.data, test_folds_list, args.seed,
                slices_list, models_parts=models_parts, window_stride= args.window_stride,min_hz_value=args.min_pitch_value)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=20, pin_memory=args.cuda)

print("######################## test ##########################")

print("best val measure {}: {}".format(args.measure, best_measure))
test_loss, test_raw_pitch, test_row_chrome, test_gpe, test_hist_5 = train_util.test(test_loader, best_pitch_model, criterion, 
                                                        device, slices_list, models_parts, args.p, VOICING_THRESHOLD=args.min_pitch_value +1)
if args.neptune:
    neptune.log_metric('test_raw_pitch', test_raw_pitch)
    neptune.log_metric('test_row_chrome', test_row_chrome)
    neptune.log_metric('test_loss', test_loss)
    neptune.log_metric('test_gpe', test_gpe)
    neptune.log_metric('test_hist_5', test_hist_5)