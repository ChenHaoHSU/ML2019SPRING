import sys

train_fpath = sys.argv[1]
output_fpath = sys.argv[2]
model_fpath = sys.argv[3]
print('# Training data : {}'.format(train_fpath))
print('# Output path   : {}'.format(output_fpath))
print('# Model         : {}'.format(model_fpath))

import numpy as np
from keras import backend as K
from keras.models import load_model
from scipy.misc import imsave
from sys import argv
from numpy import inf
import matplotlib.pyplot as plt
import math

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

model_name = model_fpath
model = load_model(model_name)
layer_dict = dict([(layer.name, layer) for layer in model.layers])
input_img = model.input

layer_name = "conv2d_1"
print("process on layer " + layer_name)
# filter_index = range(100, 200)
filter_index = [1, 2, 3, 10, 12, 15, 16, 17, 29, 30, 32, 36, 37, 38, 45, 46, 48, \
                56, 60, 65, 74, 78, 80, 83, 86, 91, 92, 95, 96, 99,\
                108, 111, 112, 113, 119, 122, 123, 124, 125, 127, 128, 129, 132, 134, 135, 139, 141, 142, 145, 146, 148, 156,\
                156, 165, 167, 171, 175, 179, 182, 184, 187, 190, 195, 197 ]

np.random.seed(0)

# for loop
random_img = np.ones((1, 48, 48, 1))
for k, f in enumerate(filter_index):
    print("process on filter " + repr(f))
    layer_output = layer_dict[layer_name].output

    loss = K.mean(layer_output[:, :, :, f])
    grads = K.gradients(loss, input_img)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([input_img], [loss, grads])

    input_img_data = np.array(random_img)

    lr = 1.
    grad_sum = 0.0
    for i in range(5000):
        loss_value, grads_value = iterate([input_img_data])
        if i < 2 and loss_value == 0.0:
            break
        grad_sum += np.square(grads_value)
        step = lr / np.sqrt(grad_sum)
        step[step == inf] = 1.0
        step = lr
        input_img_data += step * grads_value
        print("\riteration: " + repr(i) + ", current loss: " + repr(loss_value), end="", flush=True)
        # if loss_value <= 0:
        #     break
    print("", flush=True)

    img = input_img_data[0].reshape(48, 48)
    img = deprocess_image(img)
    plt.subplot(8, math.ceil(len(filter_index) / 8), k+1)
    # plt.title(repr(f))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.imshow(input_img_data[0].reshape(48, 48), cmap='gray')
    
print("save image...")
plt.savefig("%s_%s.png" % (model_name, layer_name))
plt.show()

model_name = model_fpath
model = load_model(model_name)
layer_dict = dict([(layer.name, layer) for layer in model.layers])
input_img = model.input

layer_name = "conv2d_1"
#print("process on layer " + layer_name)
filter_index = range(32)

idx = 3579
photo = x[idx].reshape(1, 48, 48, 1)

collect_layers = list()
collect_layers.append(K.function([input_img,K.learning_phase()],[layer_dict['conv2d_1'].output]))
for cnt, fn in enumerate(collect_layers):
    im = fn([photo,0])
    #fig = plt.figure(figsize=(14,8))
    nb_filter = im[0].shape[3]
    for f in filter_index:
        plt.subplot(4, 8, f+1)
        plt.title(repr(f))
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(im[0][0,f,:,:].reshape(48, 48), cmap='gray')

print("save image...")
plt.savefig("hw3_Q5_output")
plt.show()


# from sys import argv
# import numpy as np
# import seaborn as sb
# import pandas as pd
# import matplotlib.pyplot as plt
# import csv
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
# from keras.models import load_model
# from keras.utils import plot_model, np_utils
# from sklearn.metrics import confusion_matrix
# import keras.backend as K

# x = []
# y = []

# n_row = 0
# text = open(train_fpath, 'r') 
# row = csv.reader(text , delimiter=",")
# for r in row:
# 	if n_row != 0:
# 		y.append(r[0])
# 		r[1] = np.array(r[1].split(' '))
# 		r[1] = np.reshape(r[1], (1, 48, 48))
# 		x.append(r[1])
# 	n_row = n_row+1
# text.close()
# x = np.array(x)
# y = np.array(y)
# x = x.astype(np.float64)
# x = x/255
# y = y.astype(np.int)
# y = np_utils.to_categorical(y, num_classes=7)