import torch
import torch.nn as nn
import csv 
import sys
from torch.utils.data import Dataset
import numpy as np


def loaddata(is_train  ,save = False , train_file = None, test_file = None ):
    if is_train:
        train_x, train_y = [], []
        with open(train_file, 'r' , encoding = 'utf-8') as f:
            n_row = 0
            rows = csv.reader(f , delimiter=",")
            for line in rows:
                if n_row != 0:
                    train_y.append(float(line[0]))
                    train_x.append([float(i) for i in line[1].split(" ")])
                n_row += 1

        train_x = np.array(train_x)
        print(train_x.shape)
        train_x = np.reshape(train_x , (28709  , 48 , 48 ,1))
        print(train_x.shape)
        train_y = np.array(train_y, dtype = np.float)
        print(train_y.shape)
        if save == True:
            np.save("train_x.npy" , train_x)
            np.save("train_y.npy" , train_y)
        return train_x , train_y

    ##############################load testing set #################################
    else:
        test_x = []
        with open(test_file, 'r' , encoding = 'utf-8') as f:
            n_row = 0
            rows = csv.reader(f , delimiter=",")
            for line in rows:
                if n_row != 0:
                    test_x.append([float(i) for i in line[1].split(" ")])
                n_row += 1

        test_x = np.array(test_x)
        print("test_x:" , test_x.shape)
        test_x = np.reshape(test_x , (test_x.shape[0],   48 , 48 ,1 ))
        print("test_x: reshape" , test_x.shape) 
        if save == True:
            np.save("test_x.npy" , test_x)
        return test_x

class MyDataset(Dataset):
    def __init__(self, is_train ,  loadfiles= False , save=False ,  train_file = None , testx_file = None  ,mean = None , std = None , transform = None ):
        self.is_train = is_train
        self.transform = transform
        if loadfiles != False:
            if self.is_train == True:
                self.train_x = np.load(loadfiles[0])
                self.train_y = np.load(loadfiles[1])
            else :
                self.test_x = np.load(loadfiles)
                
        elif self.is_train == True:
            self.train_x , self.train_y  = loaddata(is_train = self.is_train ,save = save ,train_file= train_file)
       
        elif train_file == None and self.is_train == False:
            self.test_x  = loaddata(is_train=self.is_train, save=save, test_file=testx_file)
            
    def __len__(self):
        if self.is_train == True:
            return self.train_y.shape[0]
        else: 
            return self.test_x.shape[0]
    
    def __getitem__(self, idx):
        if self.is_train == True:
            x = torch.Tensor(self.train_x[idx])
            x =x.view(-1 , 48 , 48)
            if self.transform is not None:
                aftertransform = self.transform(x)
                return aftertransform, torch.LongTensor(np.array(self.train_y[idx]))

            return x, torch.LongTensor(np.array(self.train_y[idx]) )

        else: # for test data
            x = torch.Tensor(self.test_x[idx])
            x =x.view(-1 , 48 , 48)
            if self.transform is not None:
                aftertrans = self.transform(x)
                return aftertrans
            return x
