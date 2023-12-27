import torch
import torch.nn as nn
import torch.nn.functional as F 

class Conv1DModel(nn.Module):
    def __init__(self, dropout_p = 0):
        super(Conv1DModel, self).__init__()
        # input: 28 x 3
        # 3 channels for each of the different modalities
        # output: 12 x 8
        self.conv1 = nn.Conv1d(in_channels = 3, out_channels = 8, kernel_size = 5, stride = 2)
        # input: 12 x 8
        # output: 10 x 16
        self.conv2 = nn.Conv1d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 1)
        # input: 10 x 16
        # output: 6 x 64
        self.conv3 = nn.Conv1d(in_channels = 16, out_channels = 64, kernel_size = 5, stride = 1)
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(64 * 6, 64)
        self.fc2 = nn.Linear(64, 3)

    
    def forward(self, x):
        x = torch.tensor(x.clone().detach().requires_grad_(True), dtype=self.conv1.weight.dtype)
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = out.view(out.size(0), -1)
        if self.training:
            out = self.fc1(self.dropout(out))
            out = self.fc2(self.dropout(out))
            return out
        out = self.fc1(out)
        out = self.fc2(out)
        return out