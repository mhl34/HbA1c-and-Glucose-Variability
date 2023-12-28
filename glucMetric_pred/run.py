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
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn.functional as F
from DataProcessor import DataProcessor
from glycemicDataset import glycemicDataset
from pp5 import pp5
from Conv1DModel import Conv1DModel
from LstmModel import LstmModel
from torch.optim.lr_scheduler import StepLR

class runModel:
    def __init__(self, mainDir):
        parser = argparse.ArgumentParser()
        parser.add_argument("-m", "--modelType", dest="modelType", help="input the type of model you want to use")
        parser.add_argument("-gm", "--glucMetric", dest="glucMetric", help="input the type of glucose metric you want to regress for")
        args = parser.parse_args()
        self.modelType = args.modelType
        self.glucMetric = args.glucMetric
        self.dtype = torch.double if self.modelType == "conv1d" else torch.float64
        self.mainDir = mainDir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_norm = 1
        self.seq_length = 28
        self.num_epochs = 50
        self.dropout_p = 0.5
        self.model = self.modelChooser(self.modelType)

    def modelChooser(self, modelType):
        if modelType == "conv1d":
            return Conv1DModel(self.dropout_p)
        elif modelType == "lstm":
            return LstmModel(input_size = self.seq_length, hidden_size = 100, num_layers = 8, batch_first = True, dropout = 0.5, dtype = self.dtype)
        return None

    def train(self, samples, model):
        model.train()
        # load in classes
        dataProcessor = DataProcessor(self.mainDir)
        pp5vals = pp5()

        glucoseData = dataProcessor.loadData(samples, "dexcom")
        edaData = dataProcessor.loadData(samples, "eda")
        tempData = dataProcessor.loadData(samples, "temp")
        hrData = dataProcessor.loadData(samples, "hr")

        hba1c = dataProcessor.hba1c(samples)

        train_dataset = glycemicDataset(samples, glucoseData, edaData, hrData, tempData, hba1c, metric = self.glucMetric, dtype = self.dtype, seq_length = self.seq_length)
        # returns eda, hr, temp, then hba1c
        train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr = 1e-2, weight_decay = 1e-5)
        scheduler = StepLR(optimizer, step_size=int(self.num_epochs/5), gamma=0.1)


        for epoch in range(self.num_epochs):

            progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='batch')
            
            lossLst = []
            accLst = []

            for batch_idx, (eda, hr, temp, target) in progress_bar:
                input = torch.stack((eda, hr, temp)).permute((1,0,2)).to(self.dtype)

                output = model(input).to(self.dtype).squeeze()

                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lossLst.append(loss.item())
                accLst.append(1 - self.mape(output, target))

            scheduler.step()

            print(f"epoch {epoch} training loss: {sum(lossLst)/len(lossLst)} learning rate: {scheduler.get_lr()} training accuracy: {sum(accLst)/len(accLst)}")

    def evaluate(self, samples, model):
        model.eval()
        # load in classes
        dataProcessor = DataProcessor(self.mainDir)
        pp5vals = pp5()

        glucoseData = dataProcessor.loadData(samples, "dexcom")
        edaData = dataProcessor.loadData(samples, "eda")
        tempData = dataProcessor.loadData(samples, "temp")
        hrData = dataProcessor.loadData(samples, "hr")

        hba1c = dataProcessor.hba1c(samples)

        val_dataset = glycemicDataset(samples, glucoseData, edaData, hrData, tempData, hba1c, metric = self.glucMetric, dtype = self.dtype, seq_length = self.seq_length)
        # returns eda, hr, temp, then hba1c
        val_dataloader = DataLoader(val_dataset, batch_size = 32, shuffle = True)

        criterion = nn.MSELoss()

        with torch.no_grad():
            for epoch in range(self.num_epochs):

                progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='batch')
                
                lossLst = []
                accLst = []

                for batch_idx, (eda, hr, temp, target) in progress_bar:
                    input = torch.stack((eda, hr, temp)).permute((1,0,2)).to(self.dtype)

                    output = model(input).to(self.dtype).squeeze()

                    loss = criterion(output, target)

                    lossLst.append(loss.item())
                    accLst.append(1 - self.mape(output, target))

                print(f"epoch {epoch} training loss: {sum(lossLst)/len(lossLst)} training accuracy: {sum(accLst)/len(accLst)}")

    def mape(self, pred, target):
        return torch.sum(torch.div(torch.abs(target - pred), target)) / pred.size(0)

    def run(self):
        samples = [str(i).zfill(3) for i in range(1, 17)]
        trainSamples = samples[:-5]
        valSamples = samples[-5:]
        self.train(trainSamples, self.model)
        self.evaluate(valSamples, self.model)

if __name__ == "__main__":
    mainDir = "/home/mhl34/big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.0/"
    obj = runModel(mainDir)
    obj.run()