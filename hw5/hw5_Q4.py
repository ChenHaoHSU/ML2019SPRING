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
import matplotlib.pyplot as plt
from torchvision.models import vgg16, vgg19,\
                               resnet50, resnet101,\
                               densenet121, densenet169

##########################
# User-defined Parameters
##########################
PROXY_MODEL = resnet50

input_dir = sys.argv[1]
label_fpath = 'labels.csv'
categ_fpath = 'categories.csv'
output_dir = sys.argv[2]
print('# Input dir  : {}'.format(input_dir))
print('# Label path : {}'.format(label_fpath))
print('# Categ path : {}'.format(categ_fpath))
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

def load_categ(categ_fpath):
    data = pd.read_csv(categ_fpath)
    categories = np.array(data['CategoryName'].values, dtype=str)
    return categories

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
categories = load_categ(categ_fpath)
print('# [Info] Load {} categories'.format(len(categories)))

## [2] Load pretrained model
model = PROXY_MODEL(pretrained=True)

# use eval mode
model.eval()
# loss criterion
criterion = nn.CrossEntropyLoss()

acc = 0
# id_list = [i for i in range(200)]
id_list = [103,176,189]
## [3] Add noise to each image
for i, (origin, trans) in enumerate(zip(X_origin, X_trans)):
    if i not in id_list: continue

    print('> Processing image {}'.format(i))

    # origin
    sm = torch.nn.Softmax()
    tensor_origin = transform(origin)
    tensor_origin.requires_grad = True
    zero_gradients(tensor_origin)
    output_origin = model(tensor_origin)
    output_origin = sm(output_origin)
    top3_prob_o, top3_label_o = torch.topk(output_origin, 3)

    # trans
    sm = torch.nn.Softmax()
    tensor_trans = transform(trans)
    tensor_trans.requires_grad = True
    zero_gradients(tensor_trans)
    output_trans = model(tensor_trans)
    output_trans = sm(output_trans)
    top3_prob_t, top3_label_t = torch.topk(output_trans, 3)

    # plot
    top3_label = top3_label_o[0]
    top3_prob = top3_prob_o[0]
    objects = (categories[top3_label[0]], categories[top3_label[1]], categories[top3_label[2]])
    y_pos = np.arange(len(objects))
    performance = [top3_prob[0], top3_prob[1], top3_prob[2]]
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Probability')
    plt.title('Original Image ({})'.format(i))
    plt.savefig('{}_original.png'.format(i))
    plt.close()

    # plot
    top3_label = top3_label_t[0]
    top3_prob = top3_prob_t[0]
    objects = (categories[top3_label[0]], categories[top3_label[1]], categories[top3_label[2]])
    y_pos = np.arange(len(objects))
    performance = [top3_prob[0], top3_prob[1], top3_prob[2]]
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Probability')
    plt.title('Adversarial Image ({})'.format(i))
    plt.savefig('{}_adversarial.png'.format(i))
    plt.close()

    print('Images saved.')
    