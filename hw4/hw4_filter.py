import sys
import numpy as np
import pandas as pd
from numpy import inf
from keras import backend as K
from keras.models import load_model
from scipy.misc import imsave
from sys import argv
import matplotlib.pyplot as plt
import math

train_fpath = sys.argv[1]
output_fpath = sys.argv[2]
model_fpath = sys.argv[3]
output_fpath = output_fpath if output_fpath[-1] != '/' else output_fpath[:-1]
print('# Training data : {}'.format(train_fpath))
print('# Output path   : {}'.format(output_fpath))
print('# Model         : {}'.format(model_fpath))

def deprocess_image(x):
    # # normalize tensor: center on 0., ensure std is 0.1
    # x -= x.mean()
    # x /= (x.std() + 1e-5)

    # # clip to [0, 1]
    # x += 0.5
    # x = np.clip(x, 0, 1)

    # # convert to RGB array
    # x *= 255
    # x = np.clip(x, 0, 255).astype('uint8')
    return x

model_name = model_fpath
model = load_model(model_name)
layer_dict = dict([(layer.name, layer) for layer in model.layers])
input_img = model.input

layer_name = "conv2d_1"
print("process on layer " + layer_name)
# filter_index = [1, 2, 3, 12, 16, 17, 23, 29, 30, 32, 36, 37, 38, 40, 46, 48,\
#                 54, 60, 62, 64, 65, 74, 78, 80, 83, 86, 92, 95, 98,\
#                 108, 112, 113, 119, 123, 127, 132, 135, 142, 145,\
#                 156, 165, 167, 171, 175, 179, 184, 187, 195, 197]
filter_index = [1, 2, 12, 16, 17, 29, 30, 32, 36, 37, 38, 40, 46,\
                54, 62, 65, 74, 78, 80, 83, 92, 95, 98, 108, 112, 113, 119, 123, 127, 132, 142, 145]
# print(len(filter_index))

########################
# fig2_1
########################
np.random.seed(0)
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
        # if i == 0 and loss_value == 0.0:
        #     break
        grad_sum += np.square(grads_value)
        step = lr / np.sqrt(grad_sum)
        step[step == inf] = 1.0
        step = lr
        input_img_data += step * grads_value
        print("\riteration: " + repr(i) + ", current loss: " + repr(loss_value), end="", flush=True)
        if loss_value <= 0:
            break
    print("", flush=True)

    img = input_img_data[0].reshape(48, 48)
    img = deprocess_image(img)
    plt.subplot(4, math.ceil(len(filter_index) / 4), k+1)
    plt.title(repr(f))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.imshow(input_img_data[0].reshape(48, 48), cmap='gray')
    
print("save image...")
plt.savefig('{}/fig2_1.jpg'.format(output_fpath))
# plt.show()

########################
# fig2_2
########################
def load_train(train_fpath):
    normalization = False
    data = pd.read_csv(train_fpath)
    Y_train = np.array(data['label'].values, dtype=int)
    X_train = []
    for features in data['feature'].values:
        split_features = [ int(i) for i in features.split(' ') ]
        matrix_features = np.array(split_features).reshape(48, 48, 1)
        #matrix_features = np.array(split_features).reshape(48*48)
        X_train.append(matrix_features)
    if normalization == True:
        X_train = np.array(X_train, dtype=float) / 255.0
    else:
        X_train = np.array(X_train, dtype=float)
    return X_train, Y_train

idx = 1061
x, _ = load_train(train_fpath)
photo = x[idx].reshape(1, 48, 48, 1)

# j = 0
# for i in range(1000, 1100):
#     photo = x[i].reshape(1, 48, 48, 1)
#     plt.subplot(10, 10, j+1)
#     j += 1
#     plt.title(repr(i))
#     plt.xticks([], [])
#     plt.yticks([], [])
#     plt.imshow(photo.reshape(48, 48), cmap='gray')
# plt.savefig('photo.png')

collect_layers = list()
collect_layers.append(K.function([input_img,K.learning_phase()],[layer_dict[layer_name].output]))
for cnt, fn in enumerate(collect_layers):
    im = fn([photo,0])
    #fig = plt.figure(figsize=(14,8))
    nb_filter = im[0].shape[3]
    for i, f in enumerate(filter_index):
        plt.subplot(4, math.ceil(len(filter_index) / 4), i+1)
        plt.title(repr(f))
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(im[0][0,:,:,f].reshape(48, 48), cmap='gray')

print("save image...")
plt.savefig('{}/fig2_2.jpg'.format(output_fpath))
# plt.show()