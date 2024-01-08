import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, mainDir = ""):
        self.dexcomFormat = "Dexcom_{0}.csv"
        self.accFormat = "ACC_{0}.csv"
        self.foodLogFormat = "Food_Log_{0}.csv"
        self.ibiFormat = "IBI_{0}.csv"
        self.bvpFormat = "BVP_{0}.csv"
        self.edaFormat = "EDA_{0}.csv"
        self.hrFormat = "HR_{0}.csv"
        self.tempFormat = "TEMP_{0}.csv"
        self.mainDir = mainDir
      
    def loadData(self, samples, fileType):
        data = {}
        if fileType == "dexcom":
            for sample in samples:
                df = pd.read_csv(self.mainDir + sample + "/" + self.dexcomFormat.format(sample))
                lst = pd.to_numeric(df['Glucose Value (mg/dL)']).to_numpy()
                # lst = lst[~np.isnan(lst)]
                data[sample] = lst
        elif fileType == "temp":
            for sample in samples:
                df = pd.read_csv(self.mainDir + sample + "/" + self.tempFormat.format(sample))
                lst = pd.to_numeric(df[' temp']).to_numpy()
                lst = lst.astype(np.int64)
                # lst = lst[~np.isnan(lst)]
                data[sample] = lst
        elif fileType == "eda":
            for sample in samples:
                df = pd.read_csv(self.mainDir + sample + "/" + self.edaFormat.format(sample))
                lst = pd.to_numeric(df[' eda']).to_numpy()
                # lst = lst[~np.isnan(lst)]
                data[sample] = lst
        elif fileType == "hr":
            for sample in samples:
                df = pd.read_csv(self.mainDir + sample + "/" + self.hrFormat.format(sample))
                lst = pd.to_numeric(df[' hr']).to_numpy()
                # lst = lst[~np.isnan(lst)]
                data[sample] = lst
        elif fileType == "acc":
            for sample in samples:
                df = pd.read_csv(self.mainDir + sample + "/" + self.accFormat.format(sample))
                lst_x = pd.to_numeric(df[' acc_x']).to_numpy()
                lst_y = pd.to_numeric(df[' acc_y']).to_numpy()
                lst_z = pd.to_numeric(df[' acc_z']).to_numpy()
                data[sample] = np.sqrt(np.sum([np.square(lst_x), np.square(lst_y), np.square(lst_z)], axis=0))
        else:
            for sample in samples:
                data[sample] = np.array([])
        return data
    
    def persComp(self, value, persHigh, persLow):
        if value > persHigh:
            return "PersHigh"
        elif value < persLow:
            return "PersLow"
        return "PersNorm"
    
    def persValue(self, data, swData):
        persDict = {}
        for key in data.keys():
            persDict[key] = [(data[key][i], self.persComp(data[key][i], swData[key][i][0] + swData[key][i][1], swData[key][i][0] - swData[key][i][1])) for i in range(len(data[key]))]
        return persDict
    
    def hba1c(self, samples):
        d = {}
        df = pd.read_csv(self.mainDir + "Demographics.txt", sep='\t')
        for sample in samples:
            hba1c = df.loc[df['ID'] == int(sample)]['HbA1c'].item()
            if hba1c < 5.7:
                d[sample] = 1
            elif hba1c < 6.5:
                d[sample] = 2
            else:
                d[sample] = 3
        return d