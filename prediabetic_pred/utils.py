import torch
import torch.nn as nn
import torch.optim as optim
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

random.seed(42)

# function: parses date string into DateTime Object
# input: date
# output: DateTime Object
def dateParser(date):
    mainFormat = '%Y-%m-%d %H:%M:%S.%f'
    altFormat = '%Y-%m-%d %H:%M:%S'
    try:
        return datetime.datetime.strptime(date, mainFormat)
    except ValueError:
        return datetime.datetime.strptime(date, altFormat)

