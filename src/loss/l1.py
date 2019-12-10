from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class L1_loss(nn.Module):
    def __init__(self, args):
        super(L1_loss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, outputs, targets):
    	if outputs.size(1) == 3:
    		loss = self.criterion( outputs , targets ) 
    	else:
    		loss = 0
    		for i in range(len(outputs)):
    			loss += self.criterion( outputs[i] , targets )
    	return loss

