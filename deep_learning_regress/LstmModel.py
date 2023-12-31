import torch
import torch.nn as nn
import torch.nn.functional as F
from ChannelFC import ChannelFC
import random

class LstmModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 8, batch_first = True, dropout = 0.5, dtype = torch.float64, seq_len = 28):
        super(LstmModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.dtype = dtype
        self.seq_len = seq_len
        self.lstm_encoder = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, num_layers = self.num_layers, batch_first = self.batch_first, dropout = self.dropout, dtype = self.dtype)
        self.lstm_decoder = nn.LSTM(input_size = self.hidden_size, hidden_size = self.input_size, num_layers = self.num_layers, batch_first = self.batch_first, dropout = self.dropout, dtype = self.dtype)
        self.fc1 = nn.Linear(self.seq_len * 4, 64, dtype = self.dtype)
        self.fc2 = nn.Linear(64, 1, dtype = self.dtype)
        self.ssl = True
        self.mask_len = 7
        self.dropout_layer = nn.Dropout(dropout)
        self.decoder = nn.Identity()

    def forward(self, x):
        out, _ = self.lstm_encoder(x)
        out, _ = self.lstm_decoder(out)
        # out shape (32, 3, 100)
        out = out.reshape(out.size(0), -1).to(self.dtype)
        out = self.fc1(self.dropout_layer(out))
        out = self.fc2(self.dropout_layer(out))
        return out
    
    def getMasked(self, data, mask_len = 5):
        mask = torch.ones_like(data)
        _, _, seq_len = mask.shape
        index = random.randint(0, seq_len - mask_len - 1)
        mask[:,:,index:index + mask_len] = 0
        data = data * mask
        return data