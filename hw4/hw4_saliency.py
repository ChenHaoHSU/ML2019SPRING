import sys
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import np_utils
import keras.backend as K
from numpy import inf

print('### Start Saliency Map...')

# Handle argv
train_fpath = sys.argv[1]
output_fpath = sys.argv[2]
model_fpath = sys.argv[3]
output_fpath = output_fpath if output_fpath[-1] != '/' else output_fpath[:-1]
print('# Training data : {}'.format(train_fpath))
print('# Output path   : {}'.format(output_fpath))
print('# Model         : {}'.format(model_fpath))

label = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'sad', 5:'surprised', 6:'neutral'}

# Load data
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

print('# Loading data...')
X_train, Y_train = load_train(train_fpath)
Y_train = np_utils.to_categorical(Y_train, 7)

# Load model
print('# Loading model...')
model = load_model(model_fpath)
input_img = model.input
image_ids = [15, 299, 9, 25, 70, 81, 94]

for i, id in enumerate(image_ids):

    print('# Plotting {} ({})...'.format(label[i], id))

    # Get function
    img = X_train[id].reshape(1, 48, 48, 1)
    pred = model.predict(img).argmax(axis=-1)
    target = K.mean(model.output[:, pred[0]])
    grads = K.gradients(target, input_img)[0]
    fn = K.function([input_img, K.learning_phase()], [grads])

    # Calculate saliency map
    sliency_map = fn([img, 0])[0].reshape(48, 48)
    std, mean = np.std(sliency_map), np.mean(sliency_map)
    std, mean = sliency_map.std(), sliency_map.mean()
    heaptmap = (sliency_map - mean) / (std+1e-5)
    sliency_map -= sliency_map.mean()
    sliency_map /= sliency_map.std()

    # Plot original figure
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(12, 3), ncols=3)
    original = img.reshape(48, 48)
    img1 = ax1.imshow(original, cmap='gray')
    ax1.set_title('{}. {} ({})'.format(i, label[i], id))

    thres = sliency_map.std()
    original[np.where(abs(sliency_map) <= thres)] = original.mean()

    # Plot sliency
    img2 = ax2.imshow(sliency_map, cmap='jet')
    fig.colorbar(img2, ax=ax2)
    ax2.set_title('Saliency Map'.format(label[i], id))
    plt.tight_layout()

    # Plot mask
    img3 = ax3.imshow(original, cmap='gray')
    fig.colorbar(img3, ax=ax3)
    ax3.set_title('Mask'.format(label[i], id))
    plt.tight_layout()
    plt.savefig('{}/fig1_{}.jpg'.format(output_fpath, i))
    # plt.show()
