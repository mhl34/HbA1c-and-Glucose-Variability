import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import math
import numpy as np
import random
import datetime
import sys
import os
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

class TransformerModel(nn.Module):
    # Constructor
    def __init__(
        self,
        num_features,
        num_head,
        seq_length,
        dropout_p,
        norm_first,
        dtype):
        super(TransformerModel, self).__init__()

        # INFO
        self.model_type = "Transformer"
        self.num_features = num_features
        self.num_head = num_head
        self.seq_length = seq_length
        self.dropout_p = dropout_p
        self.norm_first = norm_first
        self.dtype = dtype

        self.embedding = nn.Linear(self.seq_length, self.num_features, dtype = self.dtype)

        self.encoder_eda = nn.TransformerEncoderLayer(d_model=self.num_features, nhead=self.num_head, norm_first = True, dtype = self.dtype)
        self.encoder_hr = nn.TransformerEncoderLayer(d_model=self.num_features, nhead=self.num_head, norm_first = True, dtype = self.dtype)
        self.encoder_temp = nn.TransformerEncoderLayer(d_model=self.num_features, nhead=self.num_head, norm_first = True, dtype = self.dtype)
        self.encoder_acc = nn.TransformerEncoderLayer(d_model=self.num_features, nhead=self.num_head, norm_first = True, dtype = self.dtype)

        self.decoder = nn.TransformerDecoderLayer(d_model=self.num_features, nhead=self.num_head, norm_first = True, dtype = self.dtype)

        self.fc1 = nn.Linear(self.num_features * 4, 256, dtype = self.dtype)
        self.fc2 = nn.Linear(256, 64, dtype = self.dtype)
        self.fc3 = nn.Linear(64, 1, dtype = self.dtype)

        # nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.xavier_uniform_(self.fc2.weight)
        # nn.init.xavier_uniform_(self.fc3.weight)

        self.dropout = nn.Dropout(dropout_p)
        self.decoder = nn.Identity()

    # function: forward of model
    # input: src, tgt, tgt_mask
    # output: output after forward run through model
    def forward(self, src):
        # Src size must be (batch_size, src, sequence_length)
        # Tgt size must be (batch_size, tgt, sequence_length)
        eda = self.embedding(src[:, 0, :])
        hr = self.embedding(src[:, 1, :])
        temp = self.embedding(src[:, 2, :])
        acc = self.embedding(src[:, 3, :])

        edaTransformerOut = self.encoder_eda(eda)
        hrTransformerOut = self.encoder_hr(hr)
        tempTransformerOut = self.encoder_temp(temp)
        accTransformerOut = self.encoder_acc(acc)

        out = torch.cat((edaTransformerOut, hrTransformerOut, tempTransformerOut, accTransformerOut), 1).to(self.dtype)

        out = self.fc1(self.dropout(out))
        out = self.fc2(self.dropout(out))
        out = self.fc3(self.dropout(out))
        return out
