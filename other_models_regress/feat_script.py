import datetime
from datetime import timedelta
import pandas as pd
import dask.dataframe as dd
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import collections
from numpy.lib.stride_tricks import sliding_window_view
import scipy.stats
import time
import gc
from feat import Feat
from sklearn.model_selection import KFold

gc.set_threshold(0)

def label_gender(row):
    if row['Gender'] == 'MALE':
        return 0
    return 1

def dateParser(date):
    mainFormat = '%Y-%m-%d %H:%M:%S.%f'
    altFormat = '%Y-%m-%d %H:%M:%S'
    try:
        return datetime.datetime.strptime(date, mainFormat)
    except ValueError:
        return datetime.datetime.strptime(date, altFormat)

class featExperiment:
    def __init__(self):
        self.directory = "{0}/{1}"
        self.dexcomFormat = "Dexcom_{0}.csv"
        self.accFormat = "ACC_{0}.csv"
        self.foodLogFormat = "Food_Log_{0}.csv"
        self.ibiFormat = "IBI_{0}.csv"
        self.bvpFormat = "BVP_{0}.csv"
        self.edaFormat = "EDA_{0}.csv"
        self.hrFormat = "HR_{0}.csv"
        self.tempFormat = "TEMP_{0}.csv"
        self.random_state = 42

    def run(self):
        # set file experiment names
        samples = [str(i).zfill(3) for i in range(1,17)]
        dexcomFiles = [self.directory.format(str(i).zfill(3), self.dexcomFormat.format(str(i).zfill(3))) for i in range(1,17)]
        hrFiles = [self.directory.format(str(i).zfill(3), self.hrFormat.format(str(i).zfill(3))) for i in range(1,17)]
        bvpFiles = [self.directory.format(str(i).zfill(3), self.bvpFormat.format(str(i).zfill(3))) for i in range(1,17)]
        edaFiles = [self.directory.format(str(i).zfill(3), self.edaFormat.format(str(i).zfill(3))) for i in range(1,17)]
        tempFiles = [self.directory.format(str(i).zfill(3), self.tempFormat.format(str(i).zfill(3))) for i in range(1,17)]

        # demographics df
        demo_df = pd.read_csv("Demographics.txt", sep = '\t')
        demo_df['Gender Label'] = demo_df.apply(label_gender, axis = 1)
        hba1c = demo_df['HbA1c']
        X = demo_df.drop(columns = ['ID', 'Gender', 'HbA1c'])
        y = hba1c

        # kfold
        kf = KFold(n_splits = 3, shuffle = True, random_state = self.random_state)

        # iterate through the samples
        print("organize initial dataframe")
        d = {'id': [], 'name': [], 'time': [], 'value': []}
        # col_names = [' eda', ' temp', ' hr']
        col_names = [' eda', ' temp']
        files = {' eda': edaFiles, ' temp': tempFiles, ' hr': hrFiles}
        for i in range(16):
            sample_name = str(i + 1).zfill(3)
            for col_name in col_names:
                # get values from each file
                df = pd.read_csv(files[col_name][i])
                len_df = len(df)
                ids = [sample_name for _ in range(len_df)]
                names = [col_name[1:] for _ in range(len_df)]
                values = df[col_name].values.tolist()
                times = [dateParser(date).timestamp() for date in df['datetime'].values.tolist()]
                d['id'].extend(ids)
                d['name'].extend(names)
                d['value'].extend(values)
                d['time'].extend(times)
        zdf = pd.DataFrame.from_dict(d)
        gc.collect()

        print("create input that can be input into FEAT")
        dates = zdf.set_index(['name','id'])['time']
        values = zdf.set_index(['name','id'])['value']
        # zdf.set_index(['name','date']).to_dict(orient='tight')
        values.to_dict()
        Z = {}
        for name, zg in zdf.groupby('name'):
            values = [
                zgid['value'].values for _,zgid in zg.groupby('id')
            ]
            timestamps = [
                zgid['time'].values for _,zgid in zg.groupby('id')
            ]
            Z[name] = (values, timestamps)
        gc.collect()

        # initialize model
        print("initialize model")
        clf = Feat(max_depth = 5,
           max_dim = 5,
           gens = 10,
           pop_size = 100,
           max_time = 120, # seconds
           verbosity = 0,
           shuffle = True,
           normalize = False, # don't normalize input data
           functions = ['and','or','not','split','split_c',
                     'mean','median','max','min','variance','skew','kurtosis','slope','count'
                     ],
           backprop = True,
           batch_size = 10,
           iters = 10,
           random_state = self.random_state,
           n_jobs = 1,
           simplify = 0.1    # prune final representations
          )

        # calculate scores
        print("calculate scores")
        scores=[]

        print("split")
        for train_idx, test_idx in kf.split(X,y):
            # print('train_idx:',train_idx)
            # note that the train index is passed to FEAT's fit method
            Ztrain = {k:([v[0][i] for i in train_idx], [v[1][i] for i in train_idx]) for k,v in Z.items()}
            Ztest = {k:([v[0][i] for i in test_idx], [v[1][i] for i in test_idx]) for k,v in Z.items()}
            clf.fit(X.loc[train_idx],y.loc[train_idx],Ztrain)
            scores.append(clf.score(X.loc[test_idx],y.loc[test_idx],Ztest))

        print(f'scores: {scores}')

        print('fitting longer to all data...')
        clf.gens = 100
        clf.fit(X,y,Z)
        print(clf.get_representation())

        print('get model')
        print(clf.get_model())

        
        

if __name__ == "__main__":
    exp = featExperiment()
    exp.run()