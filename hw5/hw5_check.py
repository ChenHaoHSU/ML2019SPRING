import sys
import os
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd.gradcheck import zero_gradients
import torch.utils.data as data
from PIL import Image
from scipy.misc import imsave
from torchvision.models import vgg16, vgg19,\
                               resnet50, resnet101,\
                               densenet121, densenet169

PROXY_MODEL = resnet50

input_dir = sys.argv[1]
label_fpath = 'labels.csv'
output_dir = sys.argv[2]
print('# Input dir  : {}'.format(input_dir))
print('# Label path : {}'.format(label_fpath))
print('# Output dir : {}'.format(output_dir))

def load_input(input_dir_):
    X_train = []
    for i in range(200):
        image_file = os.path.join(input_dir_, '{:03d}.png'.format(i))
        print('\r> Loading \'{}\''.format(image_file), end="", flush=True)
        im = Image.open(image_file)
        im_arr = np.array(im.getdata()).reshape(im.size[0], im.size[1], 3)
        X_train.append(im_arr)
    print("", flush=True)
    return np.array(X_train)

def load_labels(label_fpath):
    data = pd.read_csv(label_fpath)
    Y_train = np.array(data['TrueLabel'].values, dtype=int)
    return Y_train

def transform(image):
    image = image / 255.0
    trans = transforms.Compose([transforms.ToTensor()])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image = trans(image)
    image = image.type('torch.FloatTensor')
    image = normalize(image)
    image = image.unsqueeze(0)
    return image

X_origin = load_input(input_dir)
print('# [Info] Load {} original images'.format(len(X_origin)))
X_trans = load_input(output_dir)
print('# [Info] Load {} trans images'.format(len(X_trans)))
Y_train = load_labels(label_fpath)
print('# [Info] Load {} labels'.format(len(Y_train)))

total_max = 0
DIFF_MAX = 5
invalid = []
for i, (origin, trans) in enumerate(zip(X_origin, X_trans)):
    diff_sum, diff_max = 0, 0
    assert origin.shape == trans.shape
    diff = trans - origin
    diff = np.absolute(diff)
    diff_max = np.max(diff)
    if diff_max > DIFF_MAX:
        invalid.append((i, diff_max, np.argmax(diff)))
    diff_avg = np.sum(diff) / (origin.shape[0]*origin.shape[1]*origin.shape[2])
    total_max += diff_max

print('# Invalid:', len(invalid))
print('L-inf:', total_max/200.0)

model = PROXY_MODEL(pretrained=True)
model.eval()
criterion = nn.CrossEntropyLoss()
acc_num = 0
## [3] Add noise to each image
for i, (image, target_label) in enumerate(zip(X_trans, Y_train)):
    print('\r> Checking image {}'.format(i), end="", flush=True)
    tensor_image = transform(image)
    
    # set gradients to zero
    tensor_image.requires_grad = True
    zero_gradients(tensor_image)
    
    output = model(tensor_image)
    argmax = np.argmax(output.detach().numpy())
    if argmax == target_label:
        acc_num += 1

print("", flush=True)
print('Success: {}/{} ({:2.2f}%)'.format(acc_num, 200, 100*(acc_num/200)))
print('Failure: {}/{} ({:2.2f}%)'.format((200-acc_num), 200, 100*((200-acc_num)/200)))
