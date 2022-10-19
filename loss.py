__author__ = 'YaelSegal & May Arama'

import torch.nn as nn
import torch.nn.functional as F



class MultBinaryLoss(nn.Module):
    def __init__(self, reduction='mean', lamda=1, decoders_weights= [1,2,1,1,1,1,2]):
        super(MultBinaryLoss, self).__init__()
        self.reduction = reduction

        self.decoders_weights = decoders_weights 
        # self.model_weights = [1,2,1,1,1,1,2] # MDB 
        # self.model_weights = [1,2,1,1,2] # KEELE 
        self.lamda = lamda


    def forward(self,output, target, lens):
        
        bce_loss = 0
        prec_loss = 0

        for model_idx in range(output.size(1)):

            model_pred_in_decoder = output[:, model_idx,0] # 3, max classes
            model_target_in_decoder = target[:, model_idx, 0] # 3, max classes
            if  model_idx !=0: # model_idx ==0 is the sil model
                cur_len = lens[model_idx][0]
                model_pred_bins = output[:, model_idx, 1:cur_len] # 3, max classes
                model_target_bins = target[:, model_idx, 1: cur_len]
                match = (model_target_in_decoder > 0).float().unsqueeze(1).expand(output.size(0),cur_len-1)
                bce_loss += self.decoders_weights[model_idx] * F.binary_cross_entropy(model_pred_bins * match, model_target_bins*match)

            prec_loss +=  F.mse_loss(model_pred_in_decoder, model_target_in_decoder)

 
        return  bce_loss + self.lamda * prec_loss





