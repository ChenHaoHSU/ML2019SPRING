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
trim_dir = sys.argv[3]
print('# Input dir  : {}'.format(input_dir))
print('# Label path : {}'.format(label_fpath))
print('# Output dir : {}'.format(output_dir))
print('# Trim dir   : {}'.format(trim_dir))

def load_input(input_dir_):
    X_train_ = []
    for i in range(200):
        image_file = os.path.join(input_dir_, '{:03d}.png'.format(i))
        im = Image.open(image_file)
        im_arr = np.array(im.getdata()).reshape(im.size[0], im.size[1], 3)
        X_train_.append(im_arr)
    return np.array(X_train_)

X_origin = load_input(input_dir)
print('# [Info] Load {} original images'.format(len(X_origin)))
X_trans = load_input(output_dir)
print('# [Info] Load {} trans images'.format(len(X_trans)))

total_max = 0
DIFF_MAX = 5
for i, (origin, trans) in enumerate(zip(X_origin, X_trans)):
    assert origin.shape == trans.shape
    diff = trans - origin
    # print(diff)
    clip = origin + np.clip(diff, -DIFF_MAX, DIFF_MAX)
    clip = clip.astype(np.uint8)

    output_fpath = os.path.join(trim_dir, '{:03d}.png'.format(i))
    imsave(output_fpath, clip)
    print('\r{}'.format(output_fpath), end="", flush=True)

print("", flush=True)
