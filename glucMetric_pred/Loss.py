import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.regLoss = nn.MSELoss()
        self.maskLoss = nn.MSELoss()
        self.dannLoss = nn.CrossEntropyLoss()
        self.totalLoss = None

    def forward(self, pred, target, label = None):
        if label == None:
            self.totalLoss = self.regLoss(pred, target)
        if label == "ssl":
            self.totalLoss = self.totalLoss + self.maskLoss(pred, target) 
        if label == "dann":
            self.totalLoss = self.totalLoss + self.dannLoss(pred, target)
        return self.totalLoss