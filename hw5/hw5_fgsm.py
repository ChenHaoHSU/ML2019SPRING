## [1] Import packages
import sys
import os
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torchvision.transforms as transform
from torch.autograd.gradcheck import zero_gradients
import torch.utils.data as data
from PIL import Image
from scipy.misc import imsave
from torchvision.models import vgg16, vgg19, resnet50, vgg16_bn,\
                               resnet101, densenet121, densenet169

input_dir = sys.argv[1]
label_fpath = 'labels.csv'
output_dir = sys.argv[2]
print('# Input dir  : {}'.format(input_dir))
print('# Label path : {}'.format(label_fpath))
print('# Output dir : {}'.format(output_dir))

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

def inverse_trasform(image):
    image = image * 255.0
    image = image.squeeze(0)
    image = image.transpose(0,1)
    image = image.transpose(1,2)
    return image

def write_output(images, output_dir):
    for i, im in enumerate(images):
        output_file = '{}/{:03d}.png'.format(output_dir, i)
        im.save(output_file)

X_train = load_input(input_dir)
print('# [Info] Load {} images'.format(len(X_train)))
Y_train = load_labels(label_fpath)
print('# [Info] Load {} labels'.format(len(X_train)))

## [2] Load pretrained model
model = densenet169(pretrained=True)

# use eval mode
model.eval()
# loss criterion
criterion = nn.CrossEntropyLoss()

acc_num = 0
epsilon = 0.01
## [3] Add noise to each image
for i, (image, target_label) in enumerate(zip(X_train, Y_train)):
    image = image / 255.0
    trans = transform.Compose([transform.ToTensor()])
    
    image = trans(image)
    image = image.unsqueeze(0)
    image = image.type('torch.FloatTensor')
    image.requires_grad = True
    
    # set gradients to zero
    zero_gradients(image)
    
    output = model(image)
    argmax = np.argmax(output.detach().numpy())
    if argmax == target_label:
        acc_num += 1
    
    print('\r{:03}/{:03} ({:2.2f}%)'.format(acc_num, i+1, 100*acc_num/(i+1)), end="", flush=True)

    # tensor_label = torch.LongTensor([target_label])
    # # target = target.astype(double)
    # loss = criterion(output, tensor_label)
    # loss.backward() 
    
    # # add epsilon to image
    # image = image - epsilon * image.grad.sign_()

    # # do inverse transformation
    # image = inverse_trasform(image)
    
    # # save image
    # output_fpath = os.path.join(output_dir, '{:03d}.png'.format(i))
    # mm = image.detach().numpy()
    # imsave(output_fpath, mm)

print(acc_num,'/',200)
