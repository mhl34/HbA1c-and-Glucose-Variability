import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, dtype):
        super(Loss, self).__init__()
        self.regLossFn = nn.MSELoss()
        self.maskLossFn = nn.MSELoss()
        self.dannLossFn = nn.CrossEntropyLoss()
        self.maskLoss = None
        self.dannLoss = None
        self.totalLoss = None
        self.domain_lambda = 0.001
        self.dtype = dtype

    def forward(self, pred, target, label = None):
        if label == None:
            self.totalLoss = self.regLossFn(pred, target).to(self.dtype)
        if label == "ssl":
            self.maskLoss = self.maskLossFn(pred, target)
            self.totalLoss = self.totalLoss + self.maskLoss
        if label == "dann":
            self.dannLoss = self.dannLossFn(pred, target)
            self.totalLoss = self.totalLoss + self.domain_lambda * self.dannLoss
        return self.totalLoss.to(self.dtype)