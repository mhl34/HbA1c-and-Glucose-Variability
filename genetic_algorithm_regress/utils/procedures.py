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
from utils.pp5 import pp5
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from utils.Loss import Loss
from utils.DataProcessor import DataProcessor
from utils.GeneticDataset import GeneticDataset

def model_chooser(modelType, samples):
    pass

def train(samples, model, featMetricList, main_dir, dtype = torch.float64, seq_length = 28, normalize = False, batch_size = 32, num_epochs = 20, device = "cpu"):
    print("============================")
    print("Training...")
    print("============================")
    model.train()

    # load in classes
    dataProcessor = DataProcessor(main_dir)

    glucoseData = dataProcessor.loadData(samples, "dexcom")
    edaData = dataProcessor.loadData(samples, "eda")
    tempData = dataProcessor.loadData(samples, "temp")
    hrData = dataProcessor.loadData(samples, "hr")
    accData = dataProcessor.loadData(samples, "acc")

    hba1c = dataProcessor.hba1c(samples)

    train_dataset = GeneticDataset(samples, glucoseData, edaData, hrData, tempData, accData, hba1c, featMetricList, dtype = dtype, seq_length = seq_length, normalize = normalize, device = device)
    # returns eda, hr, temp, then hba1c
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    criterion = Loss(dtype).to(device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-8)
    # optimizer = optim.SGD(model.parameters(), lr = 1e-6, momentum = 0.5, weight_decay = 1e-8)
    # scheduler = StepLR(optimizer, step_size=int(self.num_epochs/5), gamma=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):

        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch')
        
        lossLst = []
        accLst = []
        persAccList = []

        len_dataloader = len(train_dataloader)

        # sample, edaMean, hrMean, tempMean, accMean, glucPastMean, glucMean
        
        for batch_idx, features in progress_bar:
            glucPres = features.pop(-1)

            # stack the inputs and feed as 3 channel input
            input = torch.stack(features).permute((1,0,2)).to(dtype)

            target = glucPres.to(dtype)

            # zero index the dann target
            # dannTarget = torch.tensor([int(i) - 1 for i in sample]).to(torch.long)

            # for ssl
            # p = float(batch_idx + epoch * len_dataloader) / (self.num_epochs * len_dataloader)
            # alpha = 2. / (1. + np.exp(-10 * p)) - 1

            output = model(input).to(dtype).squeeze()

            # if self.modelType == "conv1d" or self.modelType == "lstm" or self.modelType == "unet":
            #     output = model(input).to(self.dtype).squeeze()
            # elif self.modelType == "transformer":
            #     output = model(target, input).to(self.dtype).squeeze()
            # elif self.modelType == "dann":
            #     modelOut = model(input, alpha)
            #     dann_output, output = modelOut[0].to(self.dtype), modelOut[1].to(self.dtype).squeeze()
            # else:
            #     modelOut = model(input)
            #     mask_output, output = modelOut[0].to(self.dtype), modelOut[1].to(self.dtype).squeeze()

            loss = criterion(output, target)

            # if self.modelType == "ssl":
            #     loss = criterion(mask_output, input, label = "ssl")
            # if self.modelType == "dann":
            #     loss = criterion(dann_output, dannTarget, label = "dann")

            optimizer.zero_grad()
            loss.backward(retain_graph = True)
            optimizer.step()

            lossLst.append(loss.item())
            # accLst.append(1 - self.mape(output, target))
            # persAccList.append(self.persAcc(output, target))
        scheduler.step()

        print(f"epoch {epoch + 1} training loss: {sum(lossLst)/len(lossLst)} learning rate: {scheduler.get_last_lr()} training accuracy: {sum(accLst)/len(accLst)}")
        
        # verify shape
        # print(output.shape, target.shape)
        # print(f"pers category accuracy: {sum(persAccList)/len(persAccList)}")
        # example output with the epoch

        for outVal, targetVal in zip(output[-1][:3], target[-1][:3]):
            print(f"output: {outVal.item()}, target: {targetVal.item()}, difference: {outVal.item() - targetVal.item()}")
    
        


def evaluate(samples, model, featMetricList, main_dir, dtype = torch.float64, seq_length = 28, normalize = False, batch_size = 32, num_epochs = 20, device = "cpu"):
    print("============================")
    print("Evaluating...")
    print("============================")
    model.eval()

    # drop auxiliary networks
    # if self.modelType == "ssl":
    #     model.decoder = nn.Identity()
    # if self.modelType == "dann":
    #     model.adversary = nn.Identity()
    # load in classes
    dataProcessor = DataProcessor(main_dir)
    pp5vals = pp5()

    glucoseData = dataProcessor.loadData(samples, "dexcom")
    edaData = dataProcessor.loadData(samples, "eda")
    tempData = dataProcessor.loadData(samples, "temp")
    hrData = dataProcessor.loadData(samples, "hr")
    accData = dataProcessor.loadData(samples, "acc")

    hba1c = dataProcessor.hba1c(samples)

    val_dataset = GeneticDataset(samples, glucoseData, edaData, hrData, tempData, accData, hba1c, featMetricList, dtype = dtype, seq_length = seq_length, normalize = normalize, device = device)
    # returns eda, hr, temp, then hba1c
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)

    criterion = Loss(dtype).to(device)

    with torch.no_grad():
        for epoch in range(num_epochs):

            progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch')
            
            lossLst = []
            accLst = []
            # persAccList = []

            len_dataloader = len(val_dataloader)
            
            # sample, edaMean, hrMean, tempMean, accMean, glucPastMean, glucMean

            for batch_idx, features in progress_bar:
                glucPres = features.pop(-1)

                # stack the inputs and feed as 3 channel input
                input = torch.stack(features).permute((1,0,2)).to(dtype)

                target = glucPres.to(dtype)

                # alpha value for dann model
                # p = float(batch_idx + epoch * len_dataloader) / (self.num_epochs * len_dataloader)
                # alpha = 2. / (1. + np.exp(-10 * p)) - 1
                
                output = model(input).to(dtype).squeeze()

                # identify what type of outputs come from the model
                # if self.modelType == "conv1d" or self.modelType == "lstm":
                #     output = model(input).to(self.dtype).squeeze()
                # elif self.modelType == "transformer":
                #     output = model(target.view(target.shape[0], 1), input).to(self.dtype).squeeze()
                # elif self.modelType == "dann":
                #     modelOut = model(input, alpha)
                #     _, output = modelOut[0], modelOut[1].to(self.dtype).squeeze()
                # else:
                #     modelOut = model(input)
                #     _, output = modelOut[0], modelOut[1].to(self.dtype).squeeze()
                
                # loss is only calculated from the main task
                loss = criterion(output, target)

                lossLst.append(loss.item())
                # accLst.append(1 - self.mape(output, target))
                # persAccList.append(self.persAcc(output, glucStats))

            print(f"epoch {epoch} training loss: {sum(lossLst)/len(lossLst)} training accuracy: {sum(accLst)/len(accLst)}")

            # print(f"pers category accuracy: {sum(persAccList)/len(persAccList)}")

            # example output with the final epoch
            for outVal, targetVal in zip(output[-1][:3], target[-1][:3]):
                print(f"output: {outVal.item()}, target: {targetVal.item()}, difference: {outVal.item() - targetVal.item()}")
    return lossLst[-1]