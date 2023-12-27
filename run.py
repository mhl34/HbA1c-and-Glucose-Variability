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
from Conv1DModel import Conv1DModel

class runModel:
    def __init__(self, mainDir):
        self.mainDir = mainDir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_norm = 1
        self.seq_length = 28
        self.num_epochs = 5

    def train(self, samples, model):
        # load in classes
        dataProcessor = DataProcessor(self.mainDir)
        pp5vals = pp5()

        glucoseData = dataProcessor.loadData(samples, "dexcom")
        edaData = dataProcessor.loadData(samples, "eda")
        tempData = dataProcessor.loadData(samples, "temp")
        hrData = dataProcessor.loadData(samples, "hr")

        hba1c = dataProcessor.hba1c(samples)

        train_dataset = glycemicDataset(samples, glucoseData, edaData, hrData, tempData, hba1c, seq_length = self.seq_length)
        # returns eda, hr, temp, then hba1c
        train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())


        for epoch in range(self.num_epochs):

            progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='batch')
            
            lossLst = []
            accLst = []

            for batch_idx, (eda, hr, temp, hba1c) in progress_bar:
                input = torch.stack((eda, hr, temp)).permute((1,0,2)).to(torch.double)
                target = hba1c

                output = model(input)

                loss = criterion(output, target)

                preds = torch.argmax(output, dim = 1)

                accuracy = (preds == target).sum().item() / len(preds)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lossLst.append(loss.item())
                accLst.append(accuracy)

            print(f"epoch {epoch} training loss: {sum(lossLst)/len(lossLst)} training accuracy: {sum(accLst)/len(accLst)}")

    def evaluate(self, samples, model):
        # load in classes
        dataProcessor = DataProcessor(self.mainDir)
        pp5vals = pp5()

        glucoseData = dataProcessor.loadData(samples, "dexcom")
        edaData = dataProcessor.loadData(samples, "eda")
        tempData = dataProcessor.loadData(samples, "temp")
        hrData = dataProcessor.loadData(samples, "hr")

        hba1c = dataProcessor.hba1c(samples)

        val_dataset = glycemicDataset(samples, glucoseData, edaData, hrData, tempData, hba1c, seq_length = self.seq_length)
        # returns eda, hr, temp, then hba1c
        val_dataloader = DataLoader(val_dataset, batch_size = 32, shuffle = True)

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for epoch in range(self.num_epochs):

                progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='batch')
                
                lossLst = []
                accLst = []

                for batch_idx, (eda, hr, temp, hba1c) in progress_bar:
                    input = torch.stack((eda, hr, temp)).permute((1,0,2)).to(torch.double)
                    target = hba1c

                    output = model(input)

                    loss = criterion(output, target)

                    preds = torch.argmax(output, dim = 1)

                    accuracy = (preds == target).sum().item() / len(preds)

                    lossLst.append(loss.item())
                    accLst.append(accuracy)


                print(f"epoch {epoch} training loss: {sum(lossLst)/len(lossLst)} training accuracy: {sum(accLst)/len(accLst)}")

    def run(self):
        model = Conv1DModel()
        samples = [str(i).zfill(3) for i in range(1, 17)]
        trainSamples = samples[:-2]
        valSamples = samples[-2:]
        self.train(trainSamples, model)
        self.evaluate(valSamples, model)

if __name__ == "__main__":
    mainDir = "/home/mhl34/big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.0/"
    obj = runModel(mainDir)
    obj.run()