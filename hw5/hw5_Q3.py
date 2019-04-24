## [1] Import packages
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

##########################
# User-defined Parameters
##########################
PROXY_MODEL = vgg16

# handle sys.argv
input_dir = sys.argv[1]
label_fpath = 'labels.csv'
print('# Input dir  : {}'.format(input_dir))
print('# Label path : {}'.format(label_fpath))

def load_input(input_dir):
    X_train = []
    for i in range(200):
        image_file = os.path.join(input_dir, '{:03d}.png'.format(i))
        im = Image.open(image_file)
        im_arr = np.array(im.getdata()).reshape(im.size[0], im.size[1], 3)
        X_train.append(im_arr)
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

# [1] load images
X_train = load_input(input_dir)
print('# [Info] {} images loaded.'.format(len(X_train)))
Y_train = load_labels(label_fpath)
print('# [Info] Load {} labels'.format(len(Y_train)))

## [2] Load pretrained model
model = PROXY_MODEL(pretrained=True)

# use eval mode
model.eval()
# loss criterion
criterion = nn.CrossEntropyLoss()

acc = 0
## [3] Add noise to each image
for i, (image, true_label) in enumerate(zip(X_train,Y_train)):
    print('\r> Processing image {}'.format(i), end="", flush=True)

    # transform the image to tensor
    tensor_image = transform(image)
    
    # set gradients to zero
    tensor_image.requires_grad = True
    zero_gradients(tensor_image)
    
    # get the target label
    output = model(tensor_image)
    current_label = np.argmax(output.detach().numpy())
    if current_label == true_label:
        acc += 1

print("", flush=True)
print('{}/{} ({})'.format(acc, 200, acc/200))

# Report Q3
# vgg16       : 173/200 (0.865)
# vgg19       : 174/200 (0.870)
# resnet50    : 200/200 (1.000)
# resnet101   : 186/200 (0.930)
# densenet121 : 185/200 (0.925)
# densenet169 : 183/200 (0.915)

