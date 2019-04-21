## [1] Import packages
import numpy as np
import torch 
import torch.nn as nn
import torchvision.transforms as transform
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
from torchvision.models import vgg16, vgg19, resnet50, \
                               resnet101, densenet121, densenet169

input_dir = sys.argv[1]
label_fpath = ''
output_dir = sys.argv[2]
print('# Input dir : {}'.format(input_dir))
print('# Output dir : {}'.format(output_dir))

def load_input(input_dir):
    input_images = []
    for i in range(200):
        image_file = '{}/{:03d}.png'.format(input_dir, i)
        im = Image.open(image_file)
        input_images.append(im)
    return input_images

def load_label(label_fpath):


def write_output(images, output_dir):
    for i, im in enumerate(images):
        output_file = '{}/{:03d}.png'.format(output_dir, i)
        im.save(output_file)

input_images = load_input(input_dir)
print('# Load {} images'.format(len(input_images)))

## [2] Load pretrained model
# using pretrain proxy model, ex. VGG16, VGG19...
model = vgg16(pretrained=True)
# # or load weights from .pt file
# model = torch.load_state_dict(...)
# use eval mode
model.eval()
# loss criterion
loss = nn.CrossEntropyLoss()

## [3] Add noise to each image
for each raw_image, target_label:
    image = read(raw_image)
    # you can do some transform to the image, ex. ToTensor()
    trans = transform.Compose([transform.ToTensor()])
    
    image = trans(image)
    image = image.unsqueeze(0)
    image.requires_grad = True
    
    # set gradients to zero
    zero_gradients(image)
    
    output = model(image)
    loss = criterion(output, target_label)
    loss.backward() 
    
    # add epsilon to image
    image = image - epsilon * image.grad.sign_()

## [4] Do inverse_transform if you did some transformation
image = inverse_trasform(image) 
image = imsave(output_file, image.numpy())


