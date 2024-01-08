import torch
import torch.nn as nn
import torch.nn.functional as F 

class LstmModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 8, batch_first = True, dropout = 0.5, dtype = torch.float64):
        super(LstmModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.dtype = dtype
        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, num_layers = self.num_layers, batch_first = self.batch_first, dropout = self.dropout, dtype = self.dtype)
        self.fc1 = nn.Linear(self.hidden_size * 3, 64, dtype = self.dtype)
        self.fc2 = nn.Linear(64, 3, dtype = self.dtype)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.reshape(out.size(0), -1).to(self.dtype)
        out = self.fc1(out)
        out = self.fc2(out)
        return out