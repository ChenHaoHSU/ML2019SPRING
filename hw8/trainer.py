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
    def __init__(self  ,model, train_dataloader = None , test_dataloader = None , validation_loader = None,prefix = 'nn'):    
        self.train_loader = train_dataloader
        self.validation_loader = validation_loader

        self.test_loader = test_dataloader  
        self.__CUDA__ = torch.cuda.is_available()

        if self.__CUDA__:
       #     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            self.model = model.cuda()
            print("using cuda")
        else:
            self.model = model.cpu()

        # define hyper parameters
        self.parameters = model.parameters()
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss = None
        self.optimizer = torch.optim.Adam(self.parameters, lr=3e-4)
        self.prefix = prefix
    def train(self , num_epochs  , is_validation = 1 , max_earlystop = 25 , ensemble = 0):
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
               # print(data.size())
                output = self.model(data)
               # print(target.squeeze().size())
               # print(output.size())
                loss = self.loss_fn(output, target.squeeze())
                #if phase == 'train':
                loss.backward()
                self.optimizer.step()
               # print(output)
               # print(output.size())
               # predict = np.argmax(output.cpu().data.numpy(), 1 )
                predict = torch.max(output, 1)[1]  
                #print(target_cpu.type())
                #predict = predict.float()
               # print(predict)
               # print(predict.shape)
               # acc = np.mean((target.cpu().numpy() == predict))
                acc = np.mean((target == predict).cpu().numpy())
               # print(acc)
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
                    earlystop = 0
                    torch.save(self.model.state_dict(), self.prefix+'-epoch-'+str(epoch)+'-ensemble'+str(ensemble+1)+'.pt')
                    tot_valid.append((epoch , np.mean(valid_acc)))
                else: 
                    earlystop+=1
                    if earlystop ==max_earlystop:
                        print("earlystopping at epoch" , epoch)
                        print("max accuracy epoch at " , tot_valid[-1][0])
                        #torch.save(self.model.state_dict(), 'nn-epoch-earlystop'+str(epoch)+'-ensemble'+str(ensemble+1)+'.pt')
                        return                       
            else:
                print("Epoch: {}, Loss: {:.4f}, Acc: {:.4f}".format(epoch, np.mean(train_loss), np.mean(train_acc) ))
            
                torch.save(self.model.state_dict(), self.prefix+'-epoch'+str(epoch)+'.pt')
            self.loss = np.mean(train_loss)
      #  if is_validation == 0:        
        torch.save(self.model.state_dict(), self.prefix+'-epoch'+str(num_epochs)+'-ensemble'+str(ensemble+1)+'.pt')
    def valid(self):
        #self.model.eval()
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
            #predict = predict.float()
            acc = np.mean((target == predict).cpu().numpy())
            valid_acc.append(acc)
        return valid_acc
    
    def test(self , ensemble = False  ):
        self.model.eval()
        predict_list = []
        for _, data_x in enumerate(self.test_loader):
            if self.__CUDA__:
                data = data_x.cuda()
            else:
                data = data_x.cpu()
        
            output = self.model(data)
            
            
            if ensemble:
                if ensemble:
                    output = output.cpu()
                output = output.detach().numpy()
                #print(output.shape)
                predict_list.append(output)
                #predict_list.concatenate((predict_list , output ) , axis = 0)
                

            else:
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




        

   
        
        
