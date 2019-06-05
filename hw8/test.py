import torch
import sys
from trainer import Trainer
from dataset import TestingDataset
from torch.utils.data import DataLoader
from model import Net
import torchvision.transforms as transforms
import numpy as np
import csv

X_test = np.load("X_test.npy")
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5]),
])
dataset = TestingDataset(test_data = X_test, transform = test_transform)
testing_loader = DataLoader(dataset, batch_size=64, shuffle=False)
X_test = X_test.reshape(-1,48,48)
for i in range(1):
    model_path = "quantized_model_v1-3.pkl"
    model = Net()
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    else:
        model.load_state_dict(torch.load(model_path))
    
    trainer = Trainer(model=model, test_dataloader=testing_loader)
    output = trainer.test()

output = torch.Tensor(output)
prediction = (torch.max(output,1)[1]).numpy()
with open(sys.argv[1], 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id','label'])
    for i in range(len(prediction)):
        value = prediction[i]
        writer.writerow([i,value])
