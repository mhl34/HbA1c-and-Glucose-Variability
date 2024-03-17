import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import scipy
from utils.pp5 import pp5

class GeneticDataset(Dataset):
    # max, min, mean, q1, q3, std, skew
    def __init__(self, samples, glucose, eda, hr, temp, acc, hba1c, featMetricList, dtype = torch.float64, seq_length = 28, normalize = False):
        self.samples = samples
        self.glucose = glucose
        self.eda = eda
        self.hr = hr
        self.temp = temp
        self.acc = acc
        self.hba1c = hba1c
        self.seq_length = seq_length
        self.pp5vals = pp5()
        self.featMetricList = featMetricList
        self.featMetricDict = self.processFeatMetricList()
        self.dtype = dtype
        self.normalize = normalize

    def __len__(self):
        return self.glucose[self.samples[0]].__len__() - self.seq_length + 1
    
    def __getitem__(self, index):
        sample = random.choice(self.samples)
        # get the different samples
        glucoseSample = self.glucose[sample]
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
        # while glucTruth or edaTruth or hrTruth or tempTruth:
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
        
        sec_dict = {"eda" : edaSec, "hr" : hrSec, "temp" : tempSec, "accSec" : accSec}
        
        sample_list = []

        for data in self.featMetricDict.keys():
            for feature in self.featMetricDict[data]:
                sample_list.append(np.array(list(map(self.featureSelect(sample), np.array_split(sec_dict[data], self.seq_length)))))
        
        # create averages across sequence length
        glucPastMean = np.array(list(map(np.mean, np.array_split(glucosePastSec, self.seq_length))))
        glucMean = np.array(list(map(np.mean, np.array_split(glucoseSec, self.seq_length))))
        
        # append the glucose values to the sequence
        sample_list.append(glucPastMean)
        sample_list.append(glucMean)
        
        # normalize if needed
        if self.normalize:
            return tuple([self.normalizeFn(seq) for seq in sample_list])
        
        #return normal
        return tuple(sample_list)
    
    # max, min, mean, q1, q3, std, skew
    def featureSelect(self, feature):
        if feature == "max":
            return np.max
        elif feature == "min":
            return np.min
        elif feature == "mean":
            return np.mean
        elif feature == "q1":
            return lambda x: np.percentile(x, 25)
        elif feature == "q3":
            return lambda x: np.percentile(x, 25)
        elif feature == "skew":
            return scipy.stats.skew
        else:
            return
            
    def processFeatMetricList(self):
        feat_metric_dict = {}
        feature_type_dict = {0: "eda", 1: "hr", 2: "temp", 3: "acc"}
        feature_select_dict = {0: "max", 1: "min", 2: "mean", 3: "q1", 4: "q3", 5: "skew"}
        for feature_type in range(4):
            for feature_calc in range(6):
                idx =  feature_type * 4 + feature_calc
                if self.featMetricList[idx]:
                    feat_metric_dict[feature_type_dict[feature_type]] = feature_select_dict[feature_calc]
        return feat_metric_dict
    
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
        pp5val = pp5val_dict[sample_type]
        return True in np.isnan(sample_array[sample_start: sample_start + 2 * pp5val * self.seq_length + 1]) or len(sample_array[sample_start: sample_start  + 2 * pp5val * self.seq_length + 1]) != 2 * pp5val * self.seq_length + 1
