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

PROXY_MODEL = resnet50
diff_max = 5
epsilon = 0.04

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

def trim_Linf(origin, trans):
    assert origin.shape == trans.shape
    diff = trans - origin
    clip = origin + np.clip(diff, -diff_max, diff_max)
    print(np.max(trans - origin))
    return clip

# [1] load images
X_train = load_input(input_dir)
print('# [Info] Load {} images'.format(len(X_train)))
Y_train = load_labels(label_fpath)
print('# [Info] Load {} labels'.format(len(X_train)))

## [2] Load pretrained model
model = PROXY_MODEL(pretrained=True)

# use eval mode
model.eval()
# loss criterion
criterion = nn.CrossEntropyLoss()

acc_num = 0
## [3] Add noise to each image
for i, (image, target_label) in enumerate(zip(X_train, Y_train)):
    tensor_image = transform(image.copy())
    
    # set gradients to zero
    tensor_image.requires_grad = True
    zero_gradients(tensor_image)
    
    output = model(tensor_image)
    argmax = np.argmax(output.detach().numpy())
    if argmax == target_label:
        acc_num += 1
    
    print('\r{:03}/{:03} ({:2.2f}%)'.format(acc_num, i+1, 100*acc_num/(i+1)), end="", flush=True)

    tensor_label = torch.LongTensor([target_label])
    loss = criterion(output, tensor_label)
    loss.backward()
    
    # add epsilon to image
    tensor_image = tensor_image + epsilon * tensor_image.grad.sign_()

    # do inverse transformation
    tensor_image = inverse_transform(tensor_image)

    # trans to numpy.ndarray
    output_image = tensor_image.detach().numpy()

    # trim L-inf
    output_image = trim_Linf(image, output_image)
    
    # save image
    output_fpath = os.path.join(output_dir, '{:03d}.png'.format(i))
    imsave(output_fpath, output_image)

print("", flush=True)
