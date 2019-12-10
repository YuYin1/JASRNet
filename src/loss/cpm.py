from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CPM(nn.Module):
    def __init__(self):
        super(CPM, self).__init__()
        self.criterion = nn.MSELoss(False)

    def forward(self, outputs, targets, masks=[]):
        total_loss = 0

        for output in outputs:
            output = torch.masked_select(output , masks)
            target = torch.masked_select(targets, masks)

            stage_loss = self.criterion(output, target)
            total_loss = total_loss + stage_loss

        total_loss = total_loss / targets.size(0) / 2

        return total_loss

