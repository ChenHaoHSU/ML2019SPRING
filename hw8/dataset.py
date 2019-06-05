import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torchvision.transforms as transforms

class TrainingDataset(Dataset):
    def __init__(self, X_data, Y_data, transform=None):
        self.x_data = X_data.reshape((-1,48,48,1))
        self.y_data = Y_data
        self.transform = transform
    
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        assert idx < self.__len__()
        x = self.x_data[idx]
        x = torch.Tensor(x)
        x = x.view(-1,48,48)
        
        if self.transform is not None:
            x = self.transform(x)
        y = np.array(self.y_data[idx])
        
        return x, torch.LongTensor(y)

class TestingDataset(Dataset):
    def __init__(self, test_data, transform=None):
        self.test_data = test_data.reshape((-1,48,48,1))
        self.transform = transform
    
    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        assert idx < self.__len__()
        x = self.test_data[idx]
        x = torch.Tensor(x)
        x = x.view(-1,48,48)
        
        if self.transform is not None:
            x = self.transform(x)
        return x

