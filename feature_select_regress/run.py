import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
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
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

class runModel:
    def __init__(self, mainDir, projectDir):
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
        self.projectDir = projectDir
        self.mainDir = mainDir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_norm = 1
        self.seq_length = int(args.seq_len)
        self.num_epochs = int(args.num_epochs)
        self.normalize = args.normalize

        # model parameters
        self.dropout_p = 0.5
        self.domain_lambda = 0.01
        self.train_batch_size = 32
        self.val_batch_size = 32
        self.num_features = 4
        self.lr = 1e-3
        self.weight_decay = 1e-8

        # normalization
        self.train_mean = 0
        self.train_std = 0
        self.eps = 1e-12

    def modelChooser(self, modelType, samples):
        if modelType == "conv1d":
            print(f"model {modelType}")
            return Conv1DModel(num_features = self.num_features, dropout_p = self.dropout_p, seq_len = self.seq_length)
        elif modelType == "lstm":
            print(f"model {modelType}")
            # sweep hidden_size 16, 32, 64
            # sweep num_layers 2, 4, 8, 16
            return LstmModel(num_features = self.num_features, input_size = self.seq_length, hidden_size = 32, num_layers = 2, batch_first = True, dropout_p = self.dropout_p, dtype = self.dtype)
        elif modelType == "transformer":
            print(f"model {modelType}")
            return TransformerModel(num_features = 1024, num_head = 16, seq_length = self.seq_length, dropout_p = self.dropout_p, norm_first = True, dtype = self.dtype)
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

    def train(self, model, train_dataloader, optimizer, scheduler, criterion):
        for epoch in range(self.num_epochs):

            progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='batch')
            
            lossLst = []
            accLst = []
            persAccList = []

            len_dataloader = len(train_dataloader)

            # sample, edaMean, hrMean, tempMean, accMean, glucPastMean, glucMean
            
            # for batch_idx, (sample, acc, sugar, carb, mins, hba1c, glucPast, glucPres) in progress_bar:
            for batch_idx, data in progress_bar:
                # stack the inputs and feed as 3 channel input
                data = data.squeeze(2)
                input = data[:, :-1, :].to(self.dtype)

                target = data[:, -1, :]

                # if self.normalize:
                #     p = 2  # Using L2 norm
                #     epsilon = 1e-12
                #     input_norm = max(torch.norm(input, p=p), epsilon)  # Calculate the p-norm of v
                #     target_norm = max(torch.norm(target, p=p), epsilon)  # Find the maximum between v_norm and epsilon
                #     input = F.normalize(input)
                #     target = F.normalize(target)

                # zero index the dann target
                # dannTarget = torch.tensor([int(i) - 1 for i in sample]).to(torch.long)

                p = float(batch_idx + epoch * len_dataloader) / (self.num_epochs * len_dataloader)
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                if self.modelType == "conv1d" or self.modelType == "lstm" or self.modelType == "unet":
                    output = model(input).to(self.dtype).squeeze()
                elif self.modelType == "transformer":
                    output = model(target, input).to(self.dtype).squeeze()
                # elif self.modelType == "dann":
                #     modelOut = model(input, alpha)
                #     dann_output, output = modelOut[0].to(self.dtype), modelOut[1].to(self.dtype).squeeze()
                else:
                    modelOut = model(input)
                    mask_output, output = modelOut[0].to(self.dtype), modelOut[1].to(self.dtype).squeeze()

                loss = criterion(output, target)
                # if self.modelType == "ssl":
                #     loss = criterion(mask_output, input, label = "ssl")
                # if self.modelType == "dann":
                #     loss = criterion(dann_output, dannTarget, label = "dann")

                optimizer.zero_grad()
                loss.backward(retain_graph = True)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)
                optimizer.step()

                lossLst.append(loss.item())
                output_arr = ((output * self.train_std) + self.train_mean)
                target_arr = ((target * self.train_std) + self.train_mean)
                accLst.append(1 - self.mape(output_arr, target_arr))
                # persAccList.append(self.persAcc(output, target))
            scheduler.step()

            # for outVal, targetVal in zip(output_arr.detach().numpy()[-1], target_arr.detach().numpy()[-1]):
            #     print(f"output: {outVal.item()}, target: {targetVal.item()}, difference: {(outVal.item() - targetVal.item())}")
       

            print(f"epoch {epoch + 1} training loss: {sum(lossLst)/len(lossLst)} learning rate: {scheduler.get_last_lr()} training accuracy: {sum(accLst)/len(accLst)}")
            

            # print(f"pers category accuracy: {sum(persAccList)/len(persAccList)}")

            

    def evaluate(self, model, val_dataloader, criterion):
        
        with torch.no_grad():
            epoch = 1
            progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='batch')
                
            lossLst = []
            accLst = []
            persAccList = []

            len_dataloader = len(val_dataloader)
            
            # sample, edaMean, hrMean, tempMean, accMean, glucPastMean, glucMean

            for batch_idx, data in progress_bar:
                # stack the inputs and feed as 3 channel input
                data = data.squeeze(2)
                input = data[:, :-1, :].to(self.dtype)

                target = data[:, -1, :]

                # if self.normalize:
                #     p = 2  # Using L2 norm
                #     epsilon = 1e-12
                #     input_norm = max(torch.norm(input, p=p), epsilon)  # Calculate the p-norm of v
                #     target_norm = max(torch.norm(target, p=p), epsilon)  # Find the maximum between v_norm and epsilon
                #     input = F.normalize(input)
                #     target = F.normalize(target)

                # alpha value for dann model
                p = float(batch_idx + epoch * len_dataloader) / (self.num_epochs * len_dataloader)
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                
                # identify what type of outputs come from the model
                if self.modelType == "conv1d" or self.modelType == "lstm" or self.modelType == "unet":
                    output = model(input).to(self.dtype).squeeze()
                elif self.modelType == "transformer":
                    output = model(target.view(target.shape[0], 1), input).to(self.dtype).squeeze()
                # elif self.modelType == "dann":
                #     modelOut = model(input, alpha)
                #     _, output = modelOut[0], modelOut[1].to(self.dtype).squeeze()
                else:
                    modelOut = model(input)
                    _, output = modelOut[0], modelOut[1].to(self.dtype).squeeze()
                
                # loss is only calculated from the main task
                loss = criterion(output, target)

                lossLst.append(loss.item())
                output_arr = ((output * self.train_std) + self.train_mean)
                target_arr = ((target * self.train_std) + self.train_mean)
                accLst.append(1 - self.mape(output_arr, target_arr))
                # persAccList.append(self.persAcc(output, glucStats))

            print(f"val loss: {sum(lossLst)/len(lossLst)} val accuracy: {sum(accLst)/len(accLst)}")

            # print(f"pers category accuracy: {sum(persAccList)/len(persAccList)}")

            # example output with the epoch
            plt.clf()
            plt.grid(True)
            plt.figure(figsize=(8, 6))

            # Plot the target array
            plt.plot(target_arr.detach().numpy()[-1], label='Target')

            # Plot the output arrays (first and second arrays in the tuple)
            plt.plot(output_arr.detach().numpy()[-1], label='Output')

            # Add labels and legend
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title(f'Target vs. Output (model: {self.modelType})')
            plt.legend()

            # Save the plot as a PNG file
            plt.savefig(f'plots/{self.modelType}_output_no_gluc.png')

            # example output with the epoch
            # for outVal, targetVal in zip(output[-1], target[-1]):
            #     print(f"output: {(outVal.item() * self.train_std) + self.train_mean}, target: {(targetVal.item() * self.train_std) + self.train_mean}, difference: {(outVal.item() - targetVal.item() * self.train_std) + self.train_mean}")
             

    def mape(self, pred, target):
        return (torch.mean(torch.div(torch.abs(target - pred), torch.abs(target)))).item()

    # def persAcc(self, pred, glucStats):
    #     return torch.mean((torch.abs(pred - glucStats["mean"]) < glucStats["std"]).to(torch.float64)).item()

    def save_dataloader(self, dataloader, filename):
        save_path = os.path.join(self.mainDir, filename)
        np.savez(save_path, dataloader)

    def load_dataloader(self, filename):
        load_path = os.path.join(self.mainDir, filename)
        loaded_data = np.load(load_path)
        return loaded_data['arr_0']

    def run(self):
        samples = [str(i).zfill(3) for i in range(1, 17)]
        trainSamples = samples[:-4]
        valSamples = samples[-4:]

        model = self.modelChooser(self.modelType, samples)

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

        gluc_mean = 0
        sugar_mean = 0
        carb_mean = 0
        min_mean = 0
        hba1c_mean = 0

        gluc_std = 0
        sugar_std = 0
        carb_std = 0
        min_std = 0
        hba1c_std = 0

        for sample in samples:
            glucoseSample = glucoseData[sample]
            sugarSample, carbSample = foodData[sample]
            minSample = minData[sample]
            hba1cSample = hba1c[sample]
            # drop nan
            glucoseSample = glucoseSample[~np.isnan(glucoseSample)]
            sugarSample = sugarSample[~np.isnan(sugarSample)]
            carbSample = carbSample[~np.isnan(carbSample)]
            minSample = minSample[~np.isnan(minSample)]
            hba1cSample = hba1cSample[~np.isnan(hba1cSample)]
            # means 
            gluc_mean += glucoseSample.mean().item()
            sugar_mean += sugarSample.mean().item()
            carb_mean += carbSample.mean().item()
            min_mean += minSample.mean().item()
            hba1c_mean += hba1cSample.mean().item()
            # stds
            gluc_std += glucoseSample.std().item()
            sugar_std += sugarSample.std().item()
            carb_std += carbSample.std().item()
            min_std += minSample.std().item()
            hba1c_std += hba1cSample.std().item()
        # mean
        gluc_mean /= len(samples)
        sugar_mean /= len(samples)
        carb_mean /= len(samples)
        min_mean /= len(samples)
        hba1c_mean /= len(samples)
        # std
        gluc_std /= len(samples)
        sugar_std /= len(samples)
        carb_std /= len(samples)
        min_std /= len(samples)
        hba1c_std /= len(samples)

        self.train_mean = gluc_mean
        self.train_std = gluc_std

        mean_list = [sugar_mean, carb_mean, min_mean, hba1c_mean, gluc_mean, gluc_mean]
        std_list = [sugar_std, carb_std, min_mean, hba1c_std, gluc_std, gluc_std]
        std_list = [std + self.eps if std > self.eps else 1 for std in std_list]

        # Step 2: Define a custom transform to normalize the data
        custom_transform = transforms.Compose([
            # transforms.ToTensor(),  # Convert PIL image to Tensor
            transforms.Normalize(mean = mean_list, std = std_list)  # Normalize using mean and std
        ])

        # train_dataset = FeatureDataset(trainSamples, glucoseData, edaData, hrData, tempData, accData, foodData, minData, hba1c, metric = self.glucMetric, dtype = self.dtype, seq_length = self.seq_length, transforms = custom_transform)
        # # returns eda, hr, temp, then hba1c
        # train_dataloader = DataLoader(train_dataset, batch_size = self.train_batch_size, shuffle = True)

        # Load or create train dataloader
        train_dataloader_file = "feature_select_regress/dataloader/train_dataloader.npz"
        if os.path.isfile(os.path.join(self.projectDir, train_dataloader_file)):
            train_dataloader = self.load_dataloader(train_dataloader_file)
        else:
            # Create new train dataloader
            train_dataset = FeatureDataset(trainSamples, glucoseData, edaData, hrData, tempData, accData, foodData, minData, hba1c, metric=self.glucMetric, dtype=self.dtype, seq_length=self.seq_length, transforms=custom_transform)
            train_dataloader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True)
            self.save_dataloader(train_dataloader, train_dataloader_file)

        optimizer = optim.Adam(model.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        criterion = Loss(model_type = self.modelType)
        self.train(model, train_dataloader, optimizer, scheduler, criterion)

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

        # val_dataset = FeatureDataset(valSamples, glucoseData, edaData, hrData, tempData, accData, foodData, minData, hba1c, metric = self.glucMetric, dtype = self.dtype, seq_length = self.seq_length, transforms = custom_transform)
        # # returns eda, hr, temp, then hba1c
        # val_dataloader = DataLoader(val_dataset, batch_size = self.val_batch_size, shuffle = True)

        # Load or create validation dataloader
        val_dataloader_file = "feature_select_regress/dataloader/val_dataloader.npz"
        if os.path.isfile(os.path.join(self.projectDir, val_dataloader_file)):
            val_dataloader = self.load_dataloader(val_dataloader_file)
        else:
            # Create new validation dataloader
            val_dataset = FeatureDataset(valSamples, glucoseData, edaData, hrData, tempData, accData, foodData, minData, hba1c, metric=self.glucMetric, dtype=self.dtype, seq_length=self.seq_length, transforms=custom_transform)
            val_dataloader = DataLoader(val_dataset, batch_size=self.val_batch_size, shuffle=False)
            self.save_dataloader(val_dataloader, val_dataloader_file)

        criterion = Loss(model_type = self.modelType)

        self.evaluate(model, val_dataloader, criterion)

if __name__ == "__main__":
    mainDir = "/media/nvme1/expansion/glycemic_health_data/physionet.org/files/big-ideas-glycemic-wearable/1.1.2/"
    projectDir = "/home/mhl34/HbA1c-and-Glucose-Variability/"
    # mainDir = "/Users/matthewlee/Matthew/Work/DunnLab/big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.0/"
    obj = runModel(mainDir)
    obj.run()
