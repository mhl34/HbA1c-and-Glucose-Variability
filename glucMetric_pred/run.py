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
from TransformerModel import TransformerModel
from DannModel import DannModel
from SslModel import SslModel
from torch.optim.lr_scheduler import StepLR
from Loss import Loss

class runModel:
    def __init__(self, mainDir):
        parser = argparse.ArgumentParser()
        parser.add_argument("-m", "--modelType", dest="modelType", help="input the type of model you want to use")
        parser.add_argument("-gm", "--glucMetric", default = "mean", dest="glucMetric", help="input the type of glucose metric you want to regress for")
        parser.add_argument("-e", "--epochs", default=100, dest="num_epochs", help="input the number of epochs to run")
        args = parser.parse_args()
        self.modelType = args.modelType
        self.glucMetric = args.glucMetric
        self.dtype = torch.double if self.modelType == "conv1d" else torch.float64
        self.mainDir = mainDir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_norm = 1
        self.seq_length = 28
        self.num_epochs = int(args.num_epochs)
        self.dropout_p = 0.5
        self.domain_lambda = 0.01

    def modelChooser(self, modelType, samples):
        if modelType == "conv1d":
            print(f"model {modelType}")
            return Conv1DModel(self.dropout_p)
        elif modelType == "lstm":
            print(f"model {modelType}")
            return LstmModel(input_size = self.seq_length, hidden_size = 100, num_layers = 8, batch_first = True, dropout = self.dropout_p, dtype = self.dtype)
        elif modelType == "transformer":
            print(f"model {modelType}")
            return TransformerModel(num_features = 1024, num_head = 256, seq_length = self.seq_length, dropout_p = self.dropout_p, norm_first = True, dtype = self.dtype)
        elif modelType == "dann":
            print(f"model {modelType}")
            return DannModel(self.modelType, samples, dropoout = self.dropout_p)
        elif modelType == "ssl":
            return SslModel(mask_len = 7, dropout_p = self.dropout_p)
        return None

    def train(self, samples, model):
        model.train()
        # load in classes
        dataProcessor = DataProcessor(self.mainDir)

        glucoseData = dataProcessor.loadData(samples, "dexcom")
        edaData = dataProcessor.loadData(samples, "eda")
        tempData = dataProcessor.loadData(samples, "temp")
        hrData = dataProcessor.loadData(samples, "hr")

        hba1c = dataProcessor.hba1c(samples)

        train_dataset = glycemicDataset(samples, glucoseData, edaData, hrData, tempData, hba1c, metric = self.glucMetric, dtype = self.dtype, seq_length = self.seq_length)
        # returns eda, hr, temp, then hba1c
        train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True)

        criterion = Loss()
        optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-8)
        # optimizer = optim.SGD(model.parameters(), lr = 1e-6, momentum = 0.5, weight_decay = 1e-8)
        scheduler = StepLR(optimizer, step_size=int(self.num_epochs/5), gamma=0.1)

        for epoch in range(self.num_epochs):

            progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='batch')
            
            lossLst = []
            accLst = []

            len_dataloader = len(train_dataloader)

            for batch_idx, (sample, eda, hr, temp, target) in progress_bar:
                # stack the inputs and feed as 3 channel input
                input = torch.stack((eda, hr, temp)).permute((1,0,2)).to(self.dtype)

                # zero index the dann target
                dannTarget = torch.tensor([int(i) - 1 for i in sample]).to(torch.long)

                p = float(batch_idx + epoch * len_dataloader) / (self.num_epochs * len_dataloader)
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                if self.modelType == "conv1d" or self.modelType == "transformer" or self.modelType == "lstm":
                    output = model(input).to(self.dtype).squeeze()
                elif self.modelType == "dann":
                    modelOut = model(input, alpha)
                    output, dann_output = modelOut[0].to(self.dtype), modelOut[1].to(self.dtype).squeeze()
                else:
                    modelOut = model(input)
                    maskOut, output = modelOut[0].to(self.dtype), modelOut[1].to(self.dtype).squeeze()

                loss = criterion(output, target)
                if self.modelType == "ssl":
                    loss = criterion(maskOut, input, label = "ssl")
                if self.modelType == "dann":
                    loss = criterion(dann_output, dannTarget, label = "dann")

                optimizer.zero_grad()
                loss.backward(retain_graph = True)
                optimizer.step()

                lossLst.append(loss.item())
                accLst.append(1 - self.mape(output, target))

            scheduler.step()

            print(f"epoch {epoch + 1} training loss: {sum(lossLst)/len(lossLst)} learning rate: {scheduler.get_last_lr()} training accuracy: {sum(accLst)/len(accLst)}")

    def evaluate(self, samples, model):
        model.eval()

        model.decoder = nn.Identity()
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

        criterion = Loss()

        with torch.no_grad():
            for epoch in range(self.num_epochs):

                progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='batch')
                
                lossLst = []
                accLst = []

                len_dataloader = len(val_dataloader)

                for batch_idx, (sample, eda, hr, temp, target) in progress_bar:
                    input = torch.stack((eda, hr, temp)).permute((1,0,2)).to(self.dtype)

                    # zero index the dann target
                    dannTarget = torch.tensor([int(i) - 1 for i in sample]).to(torch.long)

                    p = float(batch_idx + epoch * len_dataloader) / (self.num_epochs * len_dataloader)
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1
                    
                    # identify what type of outputs come from the model
                    if self.modelType == "conv1d" or self.modelType == "transformer" or self.modelType == "lstm":
                        output = model(input).to(self.dtype).squeeze()
                    elif self.modelType == "dann":
                        modelOut = model(input, alpha)
                        output, _ = modelOut[0].to(self.dtype), modelOut[1].to(self.dtype).squeeze()
                    else:
                        modelOut = model(input)
                        _, output = modelOut[0].to(self.dtype), modelOut[1].to(self.dtype).squeeze()
                    
                    # loss is only calculated from the main task
                    loss = criterion(output, target)

                    lossLst.append(loss.item())
                    accLst.append(1 - self.mape(output, target))

                print(f"epoch {epoch} training loss: {sum(lossLst)/len(lossLst)} training accuracy: {sum(accLst)/len(accLst)}")

    def mape(self, pred, target):
        return (torch.mean(torch.div(torch.abs(target.view(len(target), 1) - pred), torch.abs(target.view(len(target), 1))))).item()

    def run(self):
        samples = [str(i).zfill(3) for i in range(1, 17)]
        trainSamples = samples[:-5]
        valSamples = samples[-5:]
        model = self.modelChooser(self.modelType, samples)
        self.train(trainSamples, model)
        self.evaluate(valSamples, model)

if __name__ == "__main__":
    mainDir = "/home/mhl34/big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.0/"
    obj = runModel(mainDir)
    obj.run()