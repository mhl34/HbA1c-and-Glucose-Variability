import torch
import torch.nn as nn

class SslLoss(nn.Module):
    def __init__(self, ssl):
        super(SslLoss, self).__init__()
        self.ssl = ssl

    def forward(self, pred, target, maskPred, data):
        regLoss = nn.MSELoss()
        maskLoss = nn.MSELoss()
        if not self.ssl:
            return regLoss(pred, target)
        return regLoss(pred, target) + maskLoss(maskPred, data)