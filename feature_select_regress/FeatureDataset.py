from torch.utils.data import Dataset
import random
from pp5 import pp5
import numpy as np
import torch
from utils import createGlucStats

class FeatureDataset(Dataset):
    def __init__(self, samples, glucose, eda, hr, temp, acc, food, hba1c, metric = "mean", dtype = torch.float64, seq_length = 28, normalize = False):
        self.samples = samples
        self.glucose = glucose
        self.eda = eda
        self.hr = hr
        self.temp = temp
        self.acc = acc
        self.food = food
        self.hba1c = hba1c
        self.seq_length = seq_length
        self.pp5vals = pp5()
        self.metric = metric
        self.dtype = dtype
        self.normalize = normalize

    def __len__(self):
        return self.glucose[self.samples[0]].__len__() - self.seq_length + 1
    
    def __getitem__(self, index):
        sample = random.choice(self.samples)
        # get the different samples
        glucoseSample = self.glucose[sample]
        sugarSample, carbSample = self.food[sample]
        edaSample = self.eda[sample]
        hrSample = self.hr[sample]
        tempSample = self.temp[sample]
        accSample = self.acc[sample]
        hba1cSample = self.hba1c[sample]
        
        # get the start indices for different sequences
        glucStart = index
        edaStart = glucStart * self.pp5vals.eda
        hrStart = glucStart * self.pp5vals.hr
        tempStart = glucStart * self.pp5vals.temp
        accStart = glucStart * self.pp5vals.acc
        
        # truth value for gluc, eda, hr, temp, or acc
        glucTruth = self.truthCheck(glucoseSample, glucStart, "glucose")
        edaTruth = self.truthCheck(edaSample, edaStart, "eda")
        hrTruth = self.truthCheck(hrSample, hrStart, "hr")
        tempTruth = self.truthCheck(tempSample, tempStart, "temp")
        accTruth = self.truthCheck(accSample, accStart, "acc")
        
        while glucTruth or edaTruth or hrTruth or tempTruth or accTruth:
            glucStart = random.randint(0,len(glucoseSample) - self.seq_length - 1)
            edaStart = glucStart * self.pp5vals.eda
            hrStart = glucStart * self.pp5vals.hr
            tempStart = glucStart * self.pp5vals.temp
            accStart = glucStart * self.pp5vals.acc
            glucTruth = self.truthCheck(glucoseSample, glucStart, "glucose")
            edaTruth = self.truthCheck(edaSample, edaStart, "eda")
            hrTruth = self.truthCheck(hrSample, hrStart, "hr")
            tempTruth = self.truthCheck(tempSample, tempStart, "temp")
            accTruth = self.truthCheck(accSample, accStart, "acc")
            
        glucosePastSec = glucoseSample[glucStart: glucStart + self.seq_length]
        glucoseSec = glucoseSample[glucStart + self.seq_length + 1: glucStart + 2 * self.seq_length + 1]
        edaSec = edaSample[edaStart: edaStart + self.seq_length * self.pp5vals.eda]
        hrSec = hrSample[hrStart: hrStart + self.seq_length * self.pp5vals.hr]
        tempSec = tempSample[tempStart: tempStart + self.seq_length * self.pp5vals.temp]
        accSec = accSample[accStart: accStart + self.seq_length * self.pp5vals.acc]
        # glucStats = createGlucStats(glucoseSec)
        
        # create averages across sequence length
        edaMean = np.array(list(map(np.mean, np.array_split(edaSec, self.seq_length))))
        hrMean = np.array(list(map(np.mean, np.array_split(hrSec, self.seq_length))))
        tempMean = np.array(list(map(np.mean, np.array_split(tempSec, self.seq_length))))
        accMean = np.array(list(map(np.mean, np.array_split(accSec, self.seq_length))))
        glucPastMean = np.array(list(map(np.mean, np.array_split(glucosePastSec, self.seq_length))))
        glucMean = np.array(list(map(np.mean, np.array_split(glucoseSec, self.seq_length))))
        
        # normalize if needed
        if self.normalize:
            return (sample, self.normalizeFn(edaMean), self.normalizeFn(hrMean), self.normalizeFn(tempMean), self.normalizeFn(accMean), self.normalizeFn(glucPastMean), self.normalizeFn(glucMean))
        
        #return normal
        return (sample, edaMean, hrMean, tempMean, accMean, glucPastMean, glucMean)
    
    def normalizeFn(self, data, eps = 1e-5):
        data = data[~np.isnan(data)]
        var = np.var(data)
        mean = np.mean(data)
        scaled_data = (data - mean) / np.sqrt(var + eps)
        return scaled_data
    
    def truthCheck(self, sample_array, sample_start, sample_type):
        if sample_type == "glucose":
            return True in np.isnan(sample_array[sample_start: sample_start + 2 * self.seq_length + 1]) or len(sample_array[sample_start: sample_start  + 2 * self.seq_length + 1]) != 2 * self.seq_length + 1
        pp5val_dict = {"eda": self.pp5vals.eda, "hr": self.pp5vals.hr, "temp": self.pp5vals.temp, "acc": self.pp5vals.acc}
        return True in np.isnan(sample_array[sample_start: sample_start + 2 * self.seq_length + 1]) or len(sample_array[sample_start: sample_start  + 2 * self.seq_length + 1]) != 2 * self.seq_length + 1
