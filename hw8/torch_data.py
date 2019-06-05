import torch
import torch.nn as nn
import csv 
import sys
from torch.utils.data import Dataset
import numpy as np


def get_feature2(train_x):
    #square
    continuous_feature = [0,1,3,5]
    #features = [0,2,3,4,5,61,62,63]
    #train_x = np.delete(train_x , [14,52,105] , 1)
    square = train_x[: , continuous_feature]
    train_x = np.concatenate((train_x , square**2) , axis=1)
    return train_x 

def _normalize_column_0_1(X, train=True, specified_column = None, X_min = None, X_max=None):
    # The output of the function will make the specified column of the training data 
    # from 0 to 1
    # When processing testing data, we need to normalize by the value 
    # we used for processing training, so we must save the max value of the 
    # training data
    if train:
        if specified_column == None:
            specified_column = np.arange(X.shape[1])
        length = len(specified_column)
        X_max = np.reshape(np.max(X[:, specified_column], 0), (1, length))
        X_min = np.reshape(np.min(X[:, specified_column], 0), (1, length))
    print(X_max , X_min)
    X[:, specified_column] = np.divide(np.subtract(X[:, specified_column], X_min), np.subtract(X_max, X_min))
    
    return X, X_max, X_min

def loaddata(is_train  ,save = False , train_file = None, test_file = None ):
    if is_train:
        train_x, train_y = [], []
        with open(train_file, 'r' , encoding = 'utf-8') as f:
            n_row = 0
            rows = csv.reader(f , delimiter=",")
            #print(row)
            
            for line in rows:
                if n_row != 0:
                    train_y.append(float(line[0]))
                    train_x.append([float(i) for i in line[1].split(" ")])
                   # for i in range(1 , len(line)):
                    #    train_x[-1].append(float(line[i]))
                n_row += 1

        #reshape

        #print(train_x)
        train_x = np.array(train_x)
        print(train_x.shape)
        train_x = np.reshape(train_x , (28709  , 48 , 48 ,1))
        print(train_x.shape)
        train_y = np.array(train_y, dtype = np.float)
       # train_y = np.reshape(train_y , (train_y.shape[0] , 1))
        print(train_y.shape)
        #input()
    #print(train_y.shape)

   
        #train_x = get_feature6(train_x)
        #print(train_x)
       
        
        #input()
        #input()
        if save == True:
            np.save("train_x.npy" , train_x)
            np.save("train_y.npy" , train_y)

        return train_x , train_y

    ##############################load testing set #################################
    else:
        test_x = []
        #print(mean.shape , std.shape)
        with open(test_file, 'r' , encoding = 'utf-8') as f:
            n_row = 0
            rows = csv.reader(f , delimiter=",")
            #print(row)
            
            for line in rows:
                if n_row != 0:
                    #train_y.append(line[0])
                    test_x.append([float(i) for i in line[1].split(" ")])
                   # for i in range(1 , len(line)):
                    #    train_x[-1].append(float(line[i]))
                n_row += 1

        test_x = np.array(test_x)

            

        #modeltmp = logisticmodel()
        #test_x = get_feature6(test_x)
        #test_x = modeltmp.add_bias(test_x)  
        #predict = modeltmp.sigmoid(np.dot(test_x , w))
        print("test_x:" , test_x.shape)
        test_x = np.reshape(test_x , (test_x.shape[0],   48 , 48 ,1 ))
        print("test_x: reshape" , test_x.shape) 
        if save == True:
            np.save("test_x.npy" , test_x)
        return test_x
        



class MyDataset(Dataset):
    def __init__(self, is_train ,  loadfiles= False , save=False ,  train_file = None , testx_file = None  ,mean = None , std = None , transform = None ):
        """
        let's assume the csv is as follows:
        ================================
        image_path                 label
        imgs/001.png               1     
        imgs/002.png               0     
        imgs/003.png               2     
        imgs/004.png               1     
                      .
                      .
                      .
        ================================
       	And we define a function parse_csv() that parses the csv into a list of tuples 
       	[('imgs/001.png', 1), ('imgs/002.png', 0)...]
        """
        self.is_train = is_train
        ##image transform
        self.transform = transform
        #load from file
        if loadfiles != False:
            if self.is_train == True:
                self.train_x = np.load(loadfiles[0])
                self.train_y = np.load(loadfiles[1])

            else :
                self.test_x = np.load(loadfiles)
                

        elif self.is_train == True:
            self.train_x , self.train_y  = loaddata(is_train = self.is_train ,save = save ,train_file= train_file 
                                                                                 )

       
        elif train_file == None and self.is_train == False:
            #for testing data
            self.test_x  = loaddata(is_train = self.is_train ,save = save ,test_file= testx_file )

   
            
    def __len__(self):
        #print("len: " , self.train_y.shape[0])
        if self.is_train == True:
            return self.train_y.shape[0]
        else: 
            #print("len: " , )
            return self.test_x.shape[0]
    
    def __getitem__(self, idx):
        #, label = self.label[idx]
       	#
        # imread: a function that reads an image from path
        
        #img = imread(img_path)
        
        # some operations/transformations
        if self.is_train == True:
            x = torch.Tensor(self.train_x[idx])
           # print(self.train_x[idx].shape)
            x =x.view(-1 , 48 , 48)
           # print(x.size())
            #print(self.train_x.shape)
            if self.transform is not None:
                #self.train_x[idx] = self.train_x[idx].astype(np.uint8)
                #train_x[idx] = Image.fromarray(train_x[idx].astype('uint8'), 'RGB')
                aftertransform = self.transform(x)
            #    print(aftertransform)
                #print(aftertransform.size())
                #print(aftertransform.dtype)
                return aftertransform, torch.LongTensor(np.array(self.train_y[idx]))

            return x, torch.LongTensor(np.array(self.train_y[idx]) )

        else: #for test data
            x = torch.Tensor(self.test_x[idx])
           # print(self.train_x[idx].shape)
            x =x.view(-1 , 48 , 48)
            if self.transform is not None:
                #self.train_x[idx] = self.train_x[idx].astype(np.uint8)
                aftertrans = self.transform(x)
                return aftertrans
            return x
#data = MyDataset(sys.argv[1] , sys.argv[2])
#print(len(data))
