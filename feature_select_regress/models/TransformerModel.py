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

        self.embedding_eda = nn.Linear(self.seq_length, self.num_features, dtype = self.dtype)
        self.embedding_hr = nn.Linear(self.seq_length, self.num_features, dtype = self.dtype)
        self.embedding_temp = nn.Linear(self.seq_length, self.num_features, dtype = self.dtype)
        self.embedding_acc = nn.Linear(self.seq_length, self.num_features, dtype = self.dtype)
        self.embedding_gluc = nn.Linear(self.seq_length, self.num_features, dtype = self.dtype)

        self.encoder_eda = nn.TransformerEncoderLayer(d_model=self.num_features, nhead=self.num_head, norm_first = True, dtype = self.dtype)
        self.encoder_hr = nn.TransformerEncoderLayer(d_model=self.num_features, nhead=self.num_head, norm_first = True, dtype = self.dtype)
        self.encoder_temp = nn.TransformerEncoderLayer(d_model=self.num_features, nhead=self.num_head, norm_first = True, dtype = self.dtype)
        self.encoder_acc = nn.TransformerEncoderLayer(d_model=self.num_features, nhead=self.num_head, norm_first = True, dtype = self.dtype)
        self.encoder_gluc_past = nn.TransformerEncoderLayer(d_model=self.num_features, nhead=self.num_head, norm_first = True, dtype = self.dtype)
        
        self.encoder_src = nn.TransformerEncoderLayer(d_model=self.num_features, nhead=self.num_head, norm_first = True, batch_first = True, dtype = self.dtype)

        self.decoder = nn.TransformerDecoderLayer(d_model=self.num_features, nhead=self.num_head, dtype = self.dtype)
        # self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=1)  # Using a single layer

        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(self.num_features * 5, self.num_features, dtype = self.dtype)
        self.fc2 = nn.Linear(self.num_features, self.seq_length, dtype = self.dtype)

    # function: forward of model
    # input: src, tgt, tgt_mask
    # output: output after forward run through model
    def forward(self, tgt, src):
        # Src size must be (batch_size, src, sequence_length)
        # Tgt size must be (batch_size, tgt, sequence_length)
        eda = self.embedding_eda(src[:, 0, :]).unsqueeze(1)
        hr = self.embedding_hr(src[:, 1, :]).unsqueeze(1)
        temp = self.embedding_temp(src[:, 2, :]).unsqueeze(1)
        acc = self.embedding_acc(src[:, 3, :]).unsqueeze(1)
        gluc_past = self.embedding_gluc(src[:, 4, :]).unsqueeze(1)

        edaTransformerOut = self.encoder_eda(eda)
        hrTransformerOut = self.encoder_hr(hr)
        tempTransformerOut = self.encoder_temp(temp)
        accTransformerOut = self.encoder_acc(acc)
        glucPastTransformerOut = self.encoder_gluc_past(gluc_past)

        out = torch.cat((edaTransformerOut, hrTransformerOut, tempTransformerOut, accTransformerOut, glucPastTransformerOut), -1).to(self.dtype)
        
        out = F.relu(self.fc1(self.dropout(out)))
        
        tgt = self.embedding_gluc(tgt).unsqueeze(1)
        
        out = self.decoder(tgt = tgt, memory = out, tgt_mask = self.get_tgt_mask(len(tgt)))
        out = torch.tensor(out.clone().detach().requires_grad_(True), dtype=self.fc1.weight.dtype)
        out = F.relu(self.fc2(self.dropout(out)))
        return out
    
    # function: creates a mask with 0's in bottom left of matrix
    # input: size
    # output: mask
    def get_tgt_mask(self, size) -> torch.tensor:
        mask = torch.tril(torch.ones(size,size) * float('-inf')).T
        for i in range(size):
            mask[i, i] = 0
        return mask.to(self.dtype)