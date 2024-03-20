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
from UNet import UNet
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from Loss import Loss

def model_chooser(self, modelType, samples):
    pass
        # if modelType == "conv1d":
        #     print(f"model {modelType}")
        #     return Conv1DGeneticModel(num_features = num_features, dropout_p = self.dropout_p, normalize = False, seq_len = self.seq_len, dtype = self.dtype).to(self.device)
        # elif modelType == "lstm":
        #     print(f"model {modelType}")
        #     return LstmModel(num_features, seq_len = self.seq_length, hidden_size = 100, num_layers = 8, batch_first = True, dropout = self.dropout_p, dtype = self.dtype)
        # elif modelType == "transformer":
        #     print(f"model {modelType}")
        #     return TransformerModel(num_features = 1024, num_head = 256, seq_length = self.seq_length, dropout_p = self.dropout_p, norm_first = True, dtype = self.dtype)
        # elif modelType == "dann":
        #     print(f"model {modelType}")
        #     return DannModel(self.modelType, samples, dropout = self.dropout_p, seq_len = self.seq_length)
        # elif modelType == "ssl":
        #     print(f"model {modelType}")
        #     return SslModel(mask_len = 7, dropout = self.dropout_p, seq_len = self.seq_length)
        # elif modelType == "unet":
        #     print(f"model {modelType}")
        #     return UNet(self.num_features, normalize = False, seq_len = self.seq_length)
        # return None

def train(self, samples, model):
    print(self.device)
    print("============================")
    print("Training...")
    print("============================")
    model.train()
    # load in classes
    dataProcessor = DataProcessor(self.mainDir)

    glucoseData = dataProcessor.loadData(samples, "dexcom")
    edaData = dataProcessor.loadData(samples, "eda")
    tempData = dataProcessor.loadData(samples, "temp")
    hrData = dataProcessor.loadData(samples, "hr")
    accData = dataProcessor.loadData(samples, "acc")

    hba1c = dataProcessor.hba1c(samples)

    train_dataset = glycemicDataset(samples, glucoseData, edaData, hrData, tempData, accData, hba1c, metric = self.glucMetric, dtype = self.dtype, seq_length = self.seq_length, normalize = self.normalize)
    # returns eda, hr, temp, then hba1c
    train_dataloader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle = True)

    criterion = Loss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-8)
    # optimizer = optim.SGD(model.parameters(), lr = 1e-6, momentum = 0.5, weight_decay = 1e-8)
    # scheduler = StepLR(optimizer, step_size=int(self.num_epochs/5), gamma=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)

    for epoch in range(self.num_epochs):

        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='batch')
        
        lossLst = []
        accLst = []
        persAccList = []

        len_dataloader = len(train_dataloader)

        # sample, edaMean, hrMean, tempMean, accMean, glucPastMean, glucMean
        
        for batch_idx, (sample, eda, hr, temp, acc, glucPast, glucPres) in progress_bar:
            # stack the inputs and feed as 3 channel input
            input = torch.stack((eda, hr, temp, acc, glucPast)).permute((1,0,2)).to(self.dtype)

            target = glucPres

            # zero index the dann target
            dannTarget = torch.tensor([int(i) - 1 for i in sample]).to(torch.long)

            p = float(batch_idx + epoch * len_dataloader) / (self.num_epochs * len_dataloader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            if self.modelType == "conv1d" or self.modelType == "lstm" or self.modelType == "unet":
                output = model(input).to(self.dtype).squeeze()
            elif self.modelType == "transformer":
                output = model(target, input).to(self.dtype).squeeze()
            elif self.modelType == "dann":
                modelOut = model(input, alpha)
                dann_output, output = modelOut[0].to(self.dtype), modelOut[1].to(self.dtype).squeeze()
            else:
                modelOut = model(input)
                mask_output, output = modelOut[0].to(self.dtype), modelOut[1].to(self.dtype).squeeze()

            loss = criterion(output, target)
            if self.modelType == "ssl":
                loss = criterion(mask_output, input, label = "ssl")
            if self.modelType == "dann":
                loss = criterion(dann_output, dannTarget, label = "dann")

            optimizer.zero_grad()
            loss.backward(retain_graph = True)
            optimizer.step()

            lossLst.append(loss.item())
            accLst.append(1 - self.mape(output, target))
            # persAccList.append(self.persAcc(output, target))
        scheduler.step()

        print(f"epoch {epoch + 1} training loss: {sum(lossLst)/len(lossLst)} learning rate: {scheduler.get_last_lr()} training accuracy: {sum(accLst)/len(accLst)}")
        
        # print(output.shape, target.shape)

        # print(f"pers category accuracy: {sum(persAccList)/len(persAccList)}")

        # example output with the epoch
        for outVal, targetVal in zip(output[-1][:3], target[-1][:3]):
                print(f"output: {outVal.item()}, target: {targetVal.item()}, difference: {outVal.item() - targetVal.item()}")


def train(self, samples, model):
    print(self.device)
    print("============================")
    print("Training...")
    print("============================")
    model.train()
    # load in classes
    dataProcessor = DataProcessor(self.mainDir)

    glucoseData = dataProcessor.loadData(samples, "dexcom")
    edaData = dataProcessor.loadData(samples, "eda")
    tempData = dataProcessor.loadData(samples, "temp")
    hrData = dataProcessor.loadData(samples, "hr")
    accData = dataProcessor.loadData(samples, "acc")

    hba1c = dataProcessor.hba1c(samples)

    train_dataset = glycemicDataset(samples, glucoseData, edaData, hrData, tempData, accData, hba1c, metric = self.glucMetric, dtype = self.dtype, seq_length = self.seq_length, normalize = self.normalize)
    # returns eda, hr, temp, then hba1c
    train_dataloader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle = True)

    criterion = Loss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-8)
    # optimizer = optim.SGD(model.parameters(), lr = 1e-6, momentum = 0.5, weight_decay = 1e-8)
    # scheduler = StepLR(optimizer, step_size=int(self.num_epochs/5), gamma=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)

    for epoch in range(self.num_epochs):

        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='batch')
        
        lossLst = []
        accLst = []
        persAccList = []

        len_dataloader = len(train_dataloader)

        # sample, edaMean, hrMean, tempMean, accMean, glucPastMean, glucMean
        
        for batch_idx, (sample, eda, hr, temp, acc, glucPast, glucPres) in progress_bar:
            # stack the inputs and feed as 3 channel input
            input = torch.stack((eda, hr, temp, acc, glucPast)).permute((1,0,2)).to(self.dtype)

            target = glucPres

            # zero index the dann target
            dannTarget = torch.tensor([int(i) - 1 for i in sample]).to(torch.long)

            p = float(batch_idx + epoch * len_dataloader) / (self.num_epochs * len_dataloader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            if self.modelType == "conv1d" or self.modelType == "lstm" or self.modelType == "unet":
                output = model(input).to(self.dtype).squeeze()
            elif self.modelType == "transformer":
                output = model(target, input).to(self.dtype).squeeze()
            elif self.modelType == "dann":
                modelOut = model(input, alpha)
                dann_output, output = modelOut[0].to(self.dtype), modelOut[1].to(self.dtype).squeeze()
            else:
                modelOut = model(input)
                mask_output, output = modelOut[0].to(self.dtype), modelOut[1].to(self.dtype).squeeze()

            loss = criterion(output, target)
            if self.modelType == "ssl":
                loss = criterion(mask_output, input, label = "ssl")
            if self.modelType == "dann":
                loss = criterion(dann_output, dannTarget, label = "dann")

            optimizer.zero_grad()
            loss.backward(retain_graph = True)
            optimizer.step()

            lossLst.append(loss.item())
            accLst.append(1 - self.mape(output, target))
            # persAccList.append(self.persAcc(output, target))
        scheduler.step()

        print(f"epoch {epoch + 1} training loss: {sum(lossLst)/len(lossLst)} learning rate: {scheduler.get_last_lr()} training accuracy: {sum(accLst)/len(accLst)}")
        
        # print(output.shape, target.shape)

        # print(f"pers category accuracy: {sum(persAccList)/len(persAccList)}")

        # example output with the epoch
        for outVal, targetVal in zip(output[-1][:3], target[-1][:3]):
                print(f"output: {outVal.item()}, target: {targetVal.item()}, difference: {outVal.item() - targetVal.item()}")