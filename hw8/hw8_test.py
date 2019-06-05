import numpy as np
import sys
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import *
from torch_data import MyDataset

# Fix random seeds
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# parameters
BATCH_SIZE = 256

device = 'cuda' if torch.cuda.is_available() else 'cpu'

test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5],std=[0.5]),
    ])

# Argv
test_fpath = sys.argv[1]
model_fpath = sys.argv[2]
output_fpath = sys.argv[3]
print('# [Info] Argv')
print('    - Test   : {}'.format(test_fpath))
print('    - Model  : {}'.format(model_fpath))
print('    = Output : {}'.format(output_fpath))

test_dataset = MyDataset(testx_file=test_fpath, is_train=False, save=True, transform=test_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False) 

model = MobileNet_Li28()
model.load_state_dict(torch.load(model_fpath))
model.to(device)

model.eval()
predict_list = []
for _, data in enumerate(test_loader):
    data_device = data.to(device)   
    output = model(data_device)
    predict = torch.max(output, 1)[1]
    for i in predict:
        predict_list.append(i)

print('# [Info] Output prediction: {}'.format(output_fpath))
with open(output_fpath, 'w') as f:
    f.write('id,label\n')
    for i, v in enumerate(predict_list):
        f.write('%d,%d\n' %(i, v))
print('Done!')
