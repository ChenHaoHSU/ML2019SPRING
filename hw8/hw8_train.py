import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

from models import MobileNet
from trainer import Trainer
from dataset import MyDataset

train_fpath = sys.argv[1]
weight_fpath = sys.argv[2]
print('# [Info] Argv')
print('    - Train  : {}'.format(train_fpath))
print('    = Weight : {}'.format(weight_fpath))

EPOCHS = 150
BATCH_SIZE = 64
VAL_RATIO = 0.1

# Transforms
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(50),
    transforms.RandomCrop(40),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(12),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5]),
])
val_transform = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5],std=[0.5]),
])

# Create dataset
train_dataset = MyDataset(is_train=True, filename=train_fpath, transform=train_transform)
val_dataset = MyDataset(is_train=True, filename=train_fpath, transform=val_transform)    

# Split train and val
data_len = len(train_dataset)
indices = list(range(data_len))
val_len = int(VAL_RATIO*data_len)
val_idx = np.random.choice(indices, size=val_len, replace=False)
train_idx = list(set(indices) - set(val_idx))
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, sampler = train_sampler)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

# Train
print('# [Info] Start training...')
mobile = MobileNet()
trainer = Trainer(model=mobile,
                  train_loader=train_loader,
                  val_loader=val_loader, 
                  weight_fpath=weight_fpath)
trainer.train(epochs=EPOCHS)
print('Done!')
