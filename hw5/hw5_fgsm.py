## [1] Import packages
import numpy as np
import torch 
import torch.nn as nn
import torchvision.transforms as transform
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
from torchvision.models import vgg16, vgg19, resnet50, \
                               resnet101, densenet121, densenet169

## [2] Load pretrained model
# using pretrain proxy model, ex. VGG16, VGG19...
model = ...(pretrain=True)
# or load weights from .pt file
model = torch.load_state_dict(...)
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


