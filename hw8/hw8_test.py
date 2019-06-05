import csv 
import numpy as np
import math
import sys
from torch_data import MyDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch
from trainer import trainer
from model import *
import torchvision.transforms as transforms

# Fix random seeds
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# parameters
BATCH_SIZE = 256

test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5],std=[0.5]),
            ])

def ensemble(modelname, test_loader):
    model = MobileNet_Li28()
    if torch.cuda.is_available() :
        model.load_state_dict(torch.load(modelname))
    else:
        model.load_state_dict(torch.load(modelname, map_location='cpu'))
    CNNtrainer = trainer(model=model, test_dataloader=test_loader) 
    predicts = CNNtrainer.test(ensemble= True)
    return predicts

# with open(output , 'w') as f:
#     subwriter = csv.writer(f , delimiter = ',')
#     subwriter.writerow(["id" , "label"])
#     for i in range(len(predict)):
#         subwriter.writerow([str(i) , predict[i]])

# Argv
test_fpath = sys.argv[1]
model_fpath = sys.argv[2]
output_fpath = sys.argv[3]
print('# [Info] Argv')
print('    - Test   : {}'.format(test_fpath))
print('    - Model  : {}'.format(model_fpath))
print('    = Output : {}'.format(output_fpath))

# predict_list = []
# test_dataset = MyDataset(testx_file=test_fpath, is_train=False, save=True, transform=test_transform)
# test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False) 
# predict_list.append(ensemble(modelname=model_fpath, test_loader=test_loader))
# predict_list = np.array(predict_list)
# predict_list = predict_list.mean(axis=0)
# predicts = predict_list.argmax(axis=1)
# output_predict(predict=predicts, output_fpath=sys.argv[3])

test_dataset = MyDataset(testx_file=test_fpath, is_train=False, save=True, transform=test_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False) 

model = MobileNet_Li28()
if torch.cuda.is_available() :
    model.load_state_dict(torch.load(model_fpath))
else:
    model.load_state_dict(torch.load(model_fpath, map_location='cpu'))

model.eval()
predict_list = []
for _, data_x in enumerate(test_loader):
    if torch.cuda.is_available():
        data = data_x.cuda()
    else:
        data = data_x.cpu()
        
    output = model(data)
    predict = torch.max(output, 1)[1]
    for i in predict:
        predict_list.append(i)

print('# [Info] Output prediction: {}'.format(output_fpath))
with open(output_fpath, 'w') as f:
    f.write('id,label\n')
    for i, v in enumerate(predict):
        f.write('%d,%d\n' %(i, v))
print('Done!')