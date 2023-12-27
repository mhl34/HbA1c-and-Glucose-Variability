import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import math
import numpy as np
import random
import datetime
import sys
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn.functional as F
from DataProcessor import DataProcessor
from glycemicDataset import glycemicDataset
from pp5 import pp5

class runModel:
    def __init__(self, mainDir):
        self.mainDir = mainDir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_norm = 1
        self.seq_length = 28

    def run(self):
        # load in classes
        dataProcessor = DataProcessor(self.mainDir)
        pp5vals = pp5()

        # get eda, temp, glucose, and hr data of all the samples
        samples = [str(i).zfill(3) for i in range(1, 17)]
        glucoseData = dataProcessor.loadData(samples, "dexcom")
        edaData = dataProcessor.loadData(samples, "eda")
        tempData = dataProcessor.loadData(samples, "temp")
        hrData = dataProcessor.loadData(samples, "hr")

        hba1c = dataProcessor.hba1c(samples)

        sample = '001'

        train_dataset = glycemicDataset(glucoseData['001'], edaData['001'], hrData['001'], tempData['001'], hba1c['001'], seq_length = self.seq_length)
        # returns eda, hr, temp, then hba1c
        train_dataloader = DataLoader(train_dataset, batch_size = 12, shuffle = True)

        for batch_idx, (eda, hr, temp, hba1c) in enumerate(train_dataloader):
            print(f"batch_idx: {batch_idx}, hba1c: {hba1c}")

if __name__ == "__main__":
    mainDir = "/home/mhl34/big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.0/"
    obj = runModel(mainDir)
    obj.run()