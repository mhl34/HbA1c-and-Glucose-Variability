from torch.utils.data import Dataset
from pp5 import pp5
import numpy as np
import torch

class glycemicDataset(Dataset):
    def __init__(self, eda, hr, temp, hba1c, seq_length):
        self.eda = eda
        self.hr = hr
        self.temp = temp
        self.hba1c = self.hba1c
        self.seq_length = seq_length
        self.pp5vals = pp5

    def __len__(self):
        return self.X.__len__() - self.seq_length + 1
    
    def __getitem__(self, index):
        edaTruth = True in np.isnan(self.eda[index: index + self.seq_length * self.pp5vals.eda]) or len(self.eda[index: index + self.seq_length * self.pp5vals.eda]) != self.seq_length * self.pp5vals.eda
        hrTruth = True in np.isnan(self.hr[index: index + self.seq_length * self.pp5vals.hr]) or len(self.hr[index: index + self.seq_length * self.pp5vals.hr]) != self.seq_length * self.pp5vals.hr
        tempTruth = True in np.isnan(self.temp[index: index + self.seq_length * self.pp5vals.temp]) or len(self.temp[index: index + self.seq_length * self.pp5vals.temp]) != self.seq_length * self.pp5vals.temp
        if edaTruth or  hrTruth or tempTruth:
            return (float('nan'))
        else:
            return (self.eda[index : index + self.seq_length], self.hr[index : index + self.seq_length], self.temp[index : index + self.seq_length], self.hba1c)