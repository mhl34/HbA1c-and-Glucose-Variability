from torch.utils.data import Dataset
import random
from pp5 import pp5
import numpy as np
import torch

class glycemicDataset(Dataset):
    def __init__(self, samples, glucose, eda, hr, temp, hba1c, seq_length = 28):
        self.samples = samples
        self.glucose = glucose
        self.eda = eda
        self.hr = hr
        self.temp = temp
        self.hba1c = hba1c
        self.seq_length = seq_length
        self.pp5vals = pp5()

    def __len__(self):
        return self.eda.__len__() - self.seq_length + 1
    
    def __getitem__(self, index):
        sample = random.choice(self.samples)
        edaSample = self.eda[sample]
        hrSample = self.hr[sample]
        tempSample = self.temp[sample]
        hba1cSample = self.hba1c[sample]
        glucStart = random.randint(0,len(self.glucose) - self.seq_length - 1)
        edaStart = glucStart * self.pp5vals.eda
        hrStart = glucStart * self.pp5vals.hr
        tempStart = glucStart * self.pp5vals.temp
        edaTruth = True in np.isnan(edaSample[edaStart: edaStart + self.seq_length * self.pp5vals.eda]) or len(edaSample[edaStart: edaStart + self.seq_length * self.pp5vals.eda]) != self.seq_length * self.pp5vals.eda
        hrTruth = True in np.isnan(hrSample[hrStart: hrStart + self.seq_length * self.pp5vals.hr]) or len(hrSample[hrStart: hrStart + self.seq_length * self.pp5vals.hr]) != self.seq_length * self.pp5vals.hr
        tempTruth = True in np.isnan(tempSample[tempStart: tempStart + self.seq_length * self.pp5vals.temp]) or len(tempSample[tempStart: tempStart + self.seq_length * self.pp5vals.temp]) != self.seq_length * self.pp5vals.temp
        while edaTruth or hrTruth or tempTruth:
            glucStart = random.randint(0,len(self.glucose) - self.seq_length - 1)
            edaStart = glucStart * self.pp5vals.eda
            hrStart = glucStart * self.pp5vals.hr
            tempStart = glucStart * self.pp5vals.temp
            edaTruth = True in np.isnan(edaSample[edaStart: edaStart + self.seq_length * self.pp5vals.eda]) or len(edaSample[edaStart: edaStart + self.seq_length * self.pp5vals.eda]) != self.seq_length * self.pp5vals.eda
            hrTruth = True in np.isnan(hrSample[hrStart: hrStart + self.seq_length * self.pp5vals.hr]) or len(hrSample[hrStart: hrStart + self.seq_length * self.pp5vals.hr]) != self.seq_length * self.pp5vals.hr
            tempTruth = True in np.isnan(tempSample[tempStart: tempStart + self.seq_length * self.pp5vals.temp]) or len(tempSample[tempStart: tempStart + self.seq_length * self.pp5vals.temp]) != self.seq_length * self.pp5vals.temp
        return (edaSample[edaStart: edaStart + self.seq_length * self.pp5vals.eda], hrSample[hrStart: hrStart + self.seq_length * self.pp5vals.hr], tempSample[tempStart: tempStart + self.seq_length * self.pp5vals.temp], hba1cSample)