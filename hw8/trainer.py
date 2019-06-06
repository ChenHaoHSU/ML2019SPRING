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
    def __init__(self, model, train_loader=None, val_loader=None, weight_fpath='train.weight'):    
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.weight_fpath = weight_fpath
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.parameters = model.parameters()
        self.optimizer = torch.optim.Adam(self.parameters, lr=3e-4)

    def train(self, epochs):
        max_val_acc = 0
        for epoch in range(1, epochs+1):
            start_time = time.time()
            self.model.train()
            train_loss = []
            train_acc = []
            for i, (data, label) in enumerate(self.train_loader):
                X = data.to(self.device)
                Y = label.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss_fn(output, Y.squeeze())
                loss.backward()
                self.optimizer.step()
                predict = torch.max(output, 1)[1]  
                acc = np.mean((Y == predict).cpu().numpy())
                train_acc.append(acc)
                train_loss.append(loss.item())
                progress = ('#' * int(float(i)/len(self.train_loader)*40)).ljust(40)
                print ('Epoch: %03d/%03d %2.1f sec(s) | %s |' % (epoch, epochs, (time.time() - start_time), progress), end='\r', flush=True)
            valid_acc = self._valid_acc() 
            print('Epoch: {}, loss: {:.4f}, acc: {:.4f}, val_Acc: {:.4f}'.format(epoch, np.mean(train_loss), np.mean(train_acc) , np.mean(valid_acc)))
            if np.mean(valid_acc) >= max_val_acc:
                torch.save(self.model.state_dict(), 'epoch{}_{}'.format(epoch, self.weight_fpath))
                max_val_acc = np.mean(valid_acc)
   
    def _valid_acc(self):
        val_acc = []
        for _, (data, label) in enumerate(self.val_loader):
            X = data.to(self.device)
            Y = label.to(self.device)
            output = self.model(X)
            predict = torch.max(output, 1)[1]
            acc = np.mean((Y == predict).cpu().numpy())
            val_acc.append(acc)
        return val_acc
