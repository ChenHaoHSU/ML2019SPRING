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
from scipy.ndimage import gaussian_filter
from torchvision.models import vgg16, vgg19,\
                               resnet50, resnet101,\
                               densenet121, densenet169

##########################
# User-defined Parameters
##########################

input_dir = sys.argv[1]
output_dir = sys.argv[2]
print('# Input dir  : {}'.format(input_dir))
print('# Output dir : {}'.format(output_dir))

def load_input(input_dir_):
    X_origin = []
    for i in range(200):
        image_file = os.path.join(input_dir_, '{:03d}.png'.format(i))
        print('\r> Loading \'{}\''.format(image_file), end="", flush=True)
        im = Image.open(image_file)
        im_arr = np.array(im.getdata()).reshape(im.size[0], im.size[1], 3)
        X_origin.append(im_arr)
    print("", flush=True)
    return X_origin

X_origin = load_input(input_dir)
print('# [Info] Load {} original images'.format(len(X_origin)))

for i, image in enumerate(X_origin):
    print('\r> Processing {}'.format(i), end="", flush=True)
    filter_image = gaussian_filter(image, sigma=1)
    output_fpath = os.path.join(output_dir, '{:03d}.png'.format(i))
    imsave(output_fpath, filter_image)
print("", flush=True)
print('Done!')