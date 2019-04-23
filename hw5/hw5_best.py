## [1] Import packages
import sys
import os
import random
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
from torch.autograd import Variable
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
PROXY_MODEL = resnet50
EPSILON = 0.06
ITERATIONS = 50

# handle sys.argv
input_dir = sys.argv[1]
output_dir = sys.argv[2]
print('# Input dir  : {}'.format(input_dir))
print('# Output dir : {}'.format(output_dir))

def load_input(input_dir):
    X_train = []
    for i in range(200):
        image_file = os.path.join(input_dir, '{:03d}.png'.format(i))
        im = Image.open(image_file)
        im_arr = np.array(im.getdata()).reshape(im.size[0], im.size[1], 3)
        X_train.append(im_arr)
    return np.array(X_train)

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

def inverse_transform(image):
    image = image.squeeze(0)
    normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                     std=[1/0.229, 1/0.224, 1/0.225])
    image = normalize(image)
    image = image.transpose(0,1)
    image = image.transpose(1,2)
    image = image * 255.0
    return image

# [1] load images
X_train = load_input(input_dir)
print('# [Info] {} images loaded.'.format(len(X_train)))

## [2] Load pretrained model
model = PROXY_MODEL(pretrained=True)

# use eval mode
model.eval()
# loss criterion
criterion = nn.CrossEntropyLoss()

## [3] Add noise to each image
for i, image in enumerate(X_train):
    print('\r> Processing image {}'.format(i), end="", flush=True)

    # transform the image to tensor
    tensor_image = transform(image)

    # get the true label (ground truth)
    output = model(tensor_image)
    true_label = np.argmax(output.detach().numpy())

    # set gradients to zero
    tensor_image.requires_grad = True
    zero_gradients(tensor_image)

    # iterations
    for iter in range(ITERATIONS):
        # update epsilon each iteration
        epsilon = EPSILON / (iter+1)

        # get the current label
        output = model(tensor_image)
        current_label = np.argmax(output.detach().numpy())

        # break the loop if the image has different labels
        if current_label != true_label:
            # print("\nEarly break {}".format(iter))
            break

        # calculate loss
        tensor_label = torch.LongTensor([current_label])
        loss = criterion(output, tensor_label)
        loss.backward()

        # add epsilon to image
        tensor_image = tensor_image + epsilon * tensor_image.grad.sign_()
        tensor_image = Variable(tensor_image.data, requires_grad=True)

    # do inverse transformation
    tensor_image = inverse_transform(tensor_image)

    # trans to numpy.ndarray
    output_image = tensor_image.detach().numpy()
    output_image = np.clip(output_image, 0, 255)
    output_image = output_image.astype(np.uint8)
    
    # save image
    output_fpath = os.path.join(output_dir, '{:03d}.png'.format(i))
    imsave(output_fpath, output_image)

# done
print("", flush=True)
print("Done!")
