## [1] Import packages
import sys
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torchvision.transforms as transform
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
from torchvision.models import vgg16, vgg19, resnet50, \
                               resnet101, densenet121, densenet169

input_dir = sys.argv[1]
label_fpath = 'labels.csv'
output_dir = sys.argv[2]
input_dir = input_dir if input_dir[-1] != '/' else input_dir[:-1]
output_dir = output_dir if output_dir[-1] != '/' else output_dir[:-1]
print('# Input dir  : {}'.format(input_dir))
print('# Label path : {}'.format(label_fpath))
print('# Output dir : {}'.format(output_dir))

def load_input(input_dir):
    X_train = []
    for i in range(200):
        image_file = '{}/{:03d}.png'.format(input_dir, i)
        im = Image.open(image_file)
        X_train.append(im)
    return X_train

def load_labels(label_fpath):
    data = pd.read_csv(label_fpath)
    Y_train = np.array(data['TrueLabel'].values, dtype=str)
    return Y_train
    
def write_output(images, output_dir):
    for i, im in enumerate(images):
        output_file = '{}/{:03d}.png'.format(output_dir, i)
        im.save(output_file)

X_train = load_input(input_dir)
Y_train = load_labels(label_fpath)
print('# Load {} images'.format(len(X_train)))
print(Y_train)

## [2] Load pretrained model
# using pretrain proxy model, ex. VGG16, VGG19...
model = vgg16(pretrained=True)
# or load weights from .pt file
# model = torch.load_state_dict(...)

X_train = torch.FloatTensor(X_train)
Y_train = torch.LongTensor(Y_train)
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)

print(model(X_train[0]))

# use eval mode
model.eval()
# loss criterion
criterion = nn.CrossEntropyLoss()

# ## [3] Add noise to each image
# for each raw_image, target_label:
#     image = read(raw_image)
#     # you can do some transform to the image, ex. ToTensor()
#     trans = transform.Compose([transform.ToTensor()])
    
#     image = trans(image)
#     image = image.unsqueeze(0)
#     image.requires_grad = True
    
#     # set gradients to zero
#     zero_gradients(image)
    
#     output = model(image)
#     loss = criterion(output, target_label)
#     loss.backward() 
    
#     # add epsilon to image
#     image = image - epsilon * image.grad.sign_()

# ## [4] Do inverse_transform if you did some transformation
# image = inverse_trasform(image) 
# image = imsave(output_file, image.numpy())


