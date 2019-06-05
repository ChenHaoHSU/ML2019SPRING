from torch_data import MyDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import Adam
import torch.nn as nn
import torch
import os
import numpy as np
import csv
import time
from operator import itemgetter

class trainer():
    def __init__(self, model, train_dataloader=None , test_dataloader=None , validation_loader=None):    
        self.train_loader = train_dataloader
        self.validation_loader = validation_loader

        self.test_loader = test_dataloader  
        self.__CUDA__ = torch.cuda.is_available()

        if self.__CUDA__:
            self.model = model.cuda()
            print("using cuda")
        else:
            self.model = model.cpu()

        # define hyper parameters
        self.parameters = model.parameters()
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss = None
        self.optimizer = torch.optim.Adam(self.parameters, lr=3e-4)
    def train(self, num_epochs, is_validation=1):
        tot_valid = [(0 ,0.0)]
        earlystop = 0
        for epoch in range(1 , num_epochs + 1):
            epoch_start_time = time.time()
            self.model.train()
            train_loss = []
            train_acc = []
            for i, (data_x, target) in enumerate(self.train_loader):
                if self.__CUDA__:
                    data = data_x.cuda()
                    target= target.cuda()
                else:
                    data = data_x.cpu()
                    target = target.cpu()
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target.squeeze())
                loss.backward()
                self.optimizer.step()
                predict = torch.max(output, 1)[1]  
                acc = np.mean((target == predict).cpu().numpy())
                train_acc.append(acc)
                train_loss.append(loss.item())

                progress = ('#' * int(float(i)/len(self.train_loader)*40)).ljust(40)
                print ('[%03d/%03d] %2.2f sec(s) | %s |' % (epoch, num_epochs, \
                (time.time() - epoch_start_time), progress), end='\r', flush=True)
    
            
            #validation
            if is_validation == 1:
              
                valid_acc = self.valid() 
                print("Epoch: {}, Loss: {:.4f}, Acc: {:.4f}, valid_Acc: {:.4f}".format(epoch, np.mean(train_loss), np.mean(train_acc) , np.mean(valid_acc)))
                
                #earlystopping
                if np.mean(valid_acc) > max(tot_valid,key=itemgetter(1))[1]:
                    torch.save(self.model.state_dict(), 'epoch-'+str(epoch)+'.pt')
                    tot_valid.append((epoch , np.mean(valid_acc)))
            self.loss = np.mean(train_loss)
   
    def valid(self):
        valid_acc = []
        for _, (data_x, target) in enumerate(self.validation_loader):
            if self.__CUDA__:
                data = data_x.cuda()
                target= target.cuda()
            else:
                data = data_x.cpu()
                target = target.cpu()
            output = self.model(data)
            predict = torch.max(output, 1)[1]
            acc = np.mean((target == predict).cpu().numpy())
            valid_acc.append(acc)
        return valid_acc
    
    def test(self , ensemble=False):
        self.model.eval()
        predict_list = []
        for _, data_x in enumerate(self.test_loader):
            if self.__CUDA__:
                data = data_x.cuda()
            else:
                data = data_x.cpu()
        
            output = self.model(data)
            
            
            predict = torch.max(output, 1)[1]
            for i in predict:
                predict_list.append(i)

        if ensemble :
            predict_list= np.concatenate(predict_list , axis = 0)
            return predict_list
        else:
            self.writepredict(predict = predict_list , output = "vgg.csv" )
            return
            

    def writepredict(predict , output):
    #flat_list = [item for sublist in predict for item in sublist]
        with open(output , 'w') as f:
            subwriter = csv.writer(f , delimiter = ',')
            subwriter.writerow(["id" , "label"])
            for i in range(len(predict)):
                subwriter.writerow([str(i+1) , predict[i]])




        

   
        
        
