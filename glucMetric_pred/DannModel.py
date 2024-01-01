import torch
import torch.nn as nn
import torch.nn.functional as F

# input: representations, which depend on the type of model
# output: guess on domain (Person as Domain)
# params: modelType
class DannModel(nn.Module):
    def __init__(self, modelType, samples):
        super(DannModel, self).__init__()
        self.modelType = modelType
        self.samples = samples
        self.inputDim = self.inputDimCalc(self.modelType)
        self.fc1 = nn.Linear(self.inputDim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, len(self.samples))

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        return out

    def inputDimCalc(self, modelType):
        if modelType == "conv1d":
            return 64 * 6
        if modelType == "lstm":
            return 100 * 3
        if modelType == "transformer":
            return 1024 * 3
        return -1