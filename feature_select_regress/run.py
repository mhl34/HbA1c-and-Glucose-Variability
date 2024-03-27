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
# from glycemicDataset import glycemicDataset
from FeatureDataset import FeatureDataset
from pp5 import pp5
from models.Conv1DModel import Conv1DModel
from models.LstmModel import LstmModel
from models.TransformerModel import TransformerModel
from models.DannModel import DannModel
from models.SslModel import SslModel
from models.UNet import UNet
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from Loss import Loss

class runModel:
    def __init__(self, mainDir):
        parser = argparse.ArgumentParser()
        parser.add_argument("-m", "--modelType", dest="modelType", help="input the type of model you want to use")
        parser.add_argument("-gm", "--glucMetric", default = "mean", dest="glucMetric", help="input the type of glucose metric you want to regress for")
        parser.add_argument("-e", "--epochs", default=100, dest="num_epochs", help="input the number of epochs to run")
        parser.add_argument("-n", action='store_true', dest="normalize", help="input whether or not to normalize the input sequence")
        parser.add_argument("-s", "--seq_len", default=28, dest="seq_len", help="input the sequence length to analyze")
        args = parser.parse_args()
        self.modelType = args.modelType
        self.glucMetric = args.glucMetric
        self.dtype = torch.double if self.modelType == "conv1d" else torch.float64
        self.mainDir = mainDir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_norm = 1
        self.seq_length = int(args.seq_len)
        self.num_epochs = int(args.num_epochs)
        self.normalize = args.normalize
        self.dropout_p = 0.5
        self.domain_lambda = 0.01
        self.batch_size = 32
        self.num_features = 6

    def modelChooser(self, modelType, samples):
        if modelType == "conv1d":
            print(f"model {modelType}")
            return Conv1DModel(num_features = self.num_features, dropout_p = self.dropout_p, seq_len = self.seq_length)
        elif modelType == "lstm":
            print(f"model {modelType}")
            return LstmModel(num_features = self.num_features, input_size = self.seq_length, hidden_size = 100, num_layers = 8, batch_first = True, dropout_p = self.dropout_p, dtype = self.dtype)
        elif modelType == "transformer":
            print(f"model {modelType}")
            return TransformerModel(num_features = 1024, num_head = 256, seq_length = self.seq_length, dropout_p = self.dropout_p, norm_first = True, dtype = self.dtype)
        elif modelType == "dann":
            print(f"model {modelType}")
            return DannModel(self.modelType, samples, dropout = self.dropout_p, seq_len = self.seq_length)
        elif modelType == "ssl":
            print(f"model {modelType}")
            return SslModel(mask_len = 7, dropout = self.dropout_p, seq_len = self.seq_length)
        elif modelType == "unet":
            print(f"model {modelType}")
            return UNet(self.num_features, normalize = False, seq_len = self.seq_length)
        return None

    def train(self, samples, model):
        print(self.device)
        print("============================")
        print("Training...")
        print("============================")
        model.train()
        # load in classes
        dataProcessor = DataProcessor(self.mainDir)

        foodData = dataProcessor.loadData(samples, "food")
        glucoseData = dataProcessor.loadData(samples, "dexcom")
        edaData = dataProcessor.loadData(samples, "eda")
        tempData = dataProcessor.loadData(samples, "temp")
        hrData = dataProcessor.loadData(samples, "hr")
        accData = dataProcessor.loadData(samples, "acc")

        hba1c = dataProcessor.hba1c(samples)

        minData = dataProcessor.minFromMidnight(samples)

        train_dataset = FeatureDataset(samples, glucoseData, edaData, hrData, tempData, accData, foodData, minData, hba1c, metric = self.glucMetric, dtype = self.dtype, seq_length = self.seq_length, normalize = self.normalize)
        # returns eda, hr, temp, then hba1c
        train_dataloader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle = True)

        criterion = Loss(model_type = self.modelType)
        optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-8)
        # optimizer = optim.Adagrad(model.parameters(), lr=1.0)
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
            
            for batch_idx, (sample, acc, sugar, carb, mins, hba1c, glucPast, glucPres) in progress_bar:
                # stack the inputs and feed as 3 channel input
                input = torch.stack((acc, sugar, carb, mins, hba1c, glucPast)).permute((1,0,2)).to(self.dtype)

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

                print(output)

                loss = criterion(output, target)
                if self.modelType == "ssl":
                    loss = criterion(mask_output, input, label = "ssl")
                if self.modelType == "dann":
                    loss = criterion(dann_output, dannTarget, label = "dann")

                optimizer.zero_grad()
                loss.backward(retain_graph = True)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)
                optimizer.step()

                lossLst.append(loss.item())
                accLst.append(1 - self.mape(output, target))
                # persAccList.append(self.persAcc(output, target))
            scheduler.step()

            print(f"epoch {epoch + 1} training loss: {sum(lossLst)/len(lossLst)} learning rate: {scheduler.get_last_lr()} training accuracy: {sum(accLst)/len(accLst)}")
            

            # print(f"pers category accuracy: {sum(persAccList)/len(persAccList)}")

            # example output with the epoch
            for outVal, targetVal in zip(output[-1][:3], target[-1][:3]):
                    print(f"output: {outVal.item()}, target: {targetVal.item()}, difference: {outVal.item() - targetVal.item()}")

    def evaluate(self, samples, model):
        print("============================")
        print("Evaluating...")
        print("============================")
        model.eval()

        # drop auxiliary networks
        if self.modelType == "ssl":
            model.decoder = nn.Identity()
        if self.modelType == "dann":
            model.adversary = nn.Identity()
        # load in classes
        dataProcessor = DataProcessor(self.mainDir)
        pp5vals = pp5()

        foodData = dataProcessor.loadData(samples, "food")
        glucoseData = dataProcessor.loadData(samples, "dexcom")
        edaData = dataProcessor.loadData(samples, "eda")
        tempData = dataProcessor.loadData(samples, "temp")
        hrData = dataProcessor.loadData(samples, "hr")
        accData = dataProcessor.loadData(samples, "acc")

        hba1c = dataProcessor.hba1c(samples)

        minData = dataProcessor.minFromMidnight(samples)

        val_dataset = FeatureDataset(samples, glucoseData, edaData, hrData, tempData, accData, foodData, minData, hba1c, metric = self.glucMetric, dtype = self.dtype, seq_length = self.seq_length, normalize = self.normalize)
        # returns eda, hr, temp, then hba1c
        val_dataloader = DataLoader(val_dataset, batch_size = self.batch_size, shuffle = True)

        criterion = Loss(model_type = self.modelType)

        with torch.no_grad():
            for epoch in range(self.num_epochs):

                progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='batch')
                
                lossLst = []
                accLst = []
                persAccList = []

                len_dataloader = len(val_dataloader)
                
                # sample, edaMean, hrMean, tempMean, accMean, glucPastMean, glucMean

                for batch_idx, (sample, acc, sugar, carb, mins, hba1c, glucPast, glucPres) in progress_bar:
                    # stack the inputs and feed as 3 channel input
                    input = torch.stack((acc, sugar, carb, mins, hba1c, glucPast)).permute((1,0,2)).to(self.dtype)

                    target = glucPres

                    # alpha value for dann model
                    p = float(batch_idx + epoch * len_dataloader) / (self.num_epochs * len_dataloader)
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1
                    
                    # identify what type of outputs come from the model
                    if self.modelType == "conv1d" or self.modelType == "lstm":
                        output = model(input).to(self.dtype).squeeze()
                    elif self.modelType == "transformer":
                        output = model(target.view(target.shape[0], 1), input).to(self.dtype).squeeze()
                    elif self.modelType == "dann":
                        modelOut = model(input, alpha)
                        _, output = modelOut[0], modelOut[1].to(self.dtype).squeeze()
                    else:
                        modelOut = model(input)
                        _, output = modelOut[0], modelOut[1].to(self.dtype).squeeze()
                    
                    # loss is only calculated from the main task
                    loss = criterion(output, target)

                    lossLst.append(loss.item())
                    accLst.append(1 - self.mape(output, target))
                    # persAccList.append(self.persAcc(output, glucStats))

                print(f"epoch {epoch} training loss: {sum(lossLst)/len(lossLst)} training accuracy: {sum(accLst)/len(accLst)}")

                # print(f"pers category accuracy: {sum(persAccList)/len(persAccList)}")

                # example output with the final epoch
                for outVal, targetVal in zip(output[-1][:3], target[-1][:3]):
                        print(f"output: {outVal.item()}, target: {targetVal.item()}, difference: {outVal.item() - targetVal.item()}")
                

    def mape(self, pred, target):
        return (torch.mean(torch.div(torch.abs(target - pred), torch.abs(target)))).item()

    # def persAcc(self, pred, glucStats):
    #     return torch.mean((torch.abs(pred - glucStats["mean"]) < glucStats["std"]).to(torch.float64)).item()

    def run(self):
        samples = [str(i).zfill(3) for i in range(1, 17)]
        trainSamples = samples[:-5]
        valSamples = samples[-5:]
        model = self.modelChooser(self.modelType, samples)
        self.train(trainSamples, model)
        self.evaluate(valSamples, model)

if __name__ == "__main__":
    mainDir = "/media/nvme1/expansion/glycemic_health_data/physionet.org/files/big-ideas-glycemic-wearable/1.1.2/"
    # mainDir = "/Users/matthewlee/Matthew/Work/DunnLab/big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.0/"
    obj = runModel(mainDir)
    obj.run()
