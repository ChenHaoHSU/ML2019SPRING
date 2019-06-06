import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def load_train(filename=None):
    data = pd.read_csv(filename)
    Y_train = np.array(data['label'].values, dtype=int)
    X_train = []
    for features in data['feature'].values:
        split_features = [ int(i) for i in features.split(' ') ]
        matrix_features = np.array(split_features).reshape(48, 48, 1)
        X_train.append(matrix_features)
    X_train = np.array(X_train, dtype=np.float)
    return X_train, Y_train

def load_test(filename=None):
    data = pd.read_csv(test_fpath)
    # id_test = np.array(data['id'].values, dtype=int)
    X_test = []
    for features in data['feature'].values:
        split_features = [ int(i) for i in features.split(' ') ]
        matrix_features = np.array(split_features).reshape(48, 48, 1)
        X_test.append(matrix_features)
    X_test = np.array(X_test, dtype=np.float)
    return X_test

class MyDataset(Dataset):
    def __init__(self, is_train=True, filename=None, transform=None):
        self.is_train = is_train
        self.filename = filename
        self.transform = transform
        if self.is_train == True:
            self.X_train, self.Y_train = load_train(filename=self.filename)
        else:
            self.X_test = load_test(filename=self.filename)
            
    def __len__(self):
        if self.is_train == True:
            return len(self.X_train)
        else: 
            return len(self.X_test)
    
    def __getitem__(self, idx):
        if self.is_train == True:
            x = torch.Tensor(self.X_train[idx])
            x = x.view(-1, 48, 48)
            x = x if self.transform == None else self.transform(x)
            return x, torch.LongTensor(np.array(self.Y_train[idx]))
        else:
            x = torch.Tensor(self.X_test[idx])
            x = x.view(-1, 48, 48)
            x = x if self.transform == None else self.transform(x)
            return x