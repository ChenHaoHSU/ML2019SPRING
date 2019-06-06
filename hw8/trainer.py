import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import Adam
from dataset import MyDataset
from operator import itemgetter

class Trainer():
    def __init__(self, model, train_loader=None, val_loader=None):    
        self.train_loader = train_loader
        self.val_loader = val_loader

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

    def train(self, epochs):
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
            valid_acc = self.valid() 
            print("Epoch: {}, Loss: {:.4f}, Acc: {:.4f}, valid_Acc: {:.4f}".format(epoch, np.mean(train_loss), np.mean(train_acc) , np.mean(valid_acc)))
            if np.mean(valid_acc) > max(tot_valid,key=itemgetter(1))[1]:
                torch.save(self.model.state_dict(), 'epoch-'+str(epoch)+'.pt')
                tot_valid.append((epoch , np.mean(valid_acc)))

            self.loss = np.mean(train_loss)
   
    def valid(self):
        valid_acc = []
        for _, (data_x, target) in enumerate(self.val_loader):
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
