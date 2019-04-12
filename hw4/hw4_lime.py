import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import np_utils
import keras.backend as K
from lime import lime_image
from skimage.segmentation import slic

print('### Lime...')
# Handle argv
train_fpath = sys.argv[1]
output_fpath = sys.argv[2]
model_fpath = sys.argv[3]
output_fpath = output_fpath if output_fpath[-1] != '/' else output_fpath[:-1]
print('# Training data : {}'.format(train_fpath))
print('# Output path   : {}'.format(output_fpath))
print('# Model         : {}'.format(model_fpath))

label = [0, 1, 2, 3, 4, 5, 6]
image_ids = [15, 299, 9, 25, 70, 81, 94]
np.random.seed(0)

# Load data
def load_train(train_fpath):
    normalization = False
    data = pd.read_csv(train_fpath)
    Y_label = np.array(data['label'].values, dtype=int)
    X_feature = data['feature'].values
    X_train, Y_train = [], []
    for l, id in enumerate(image_ids):
        features = X_feature[id]
        Y_train.append(label[l])
        split_features = [ [int(i),int(i),int(i)] for i in features.split(' ') ]
        matrix_features = np.array(split_features).reshape(48, 48, 3)
        X_train.append(matrix_features)
    if normalization == True:
        X_train = np.array(X_train, dtype=float) / 255.0
    else:
        X_train = np.array(X_train, dtype=float)
    Y_train = np.array(Y_train)
    return X_train, Y_train

print('# Loading data...')
X_train, Y_train = load_train(train_fpath)
Y_train = np_utils.to_categorical(Y_train, 7)

# Load model
print('# Loading model...')
model = load_model(model_fpath)

x_train_rgb = X_train

def predict(input):
    gray = np.expand_dims(input[:,:,:,0], 3)
    return model.predict(gray)

def segmentation(input):
    return slic(input, n_segments=200, compactness=100)

from skimage.segmentation import mark_boundaries
for i, idx in enumerate(image_ids):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(x_train_rgb[i], predict, segmentation_fn=segmentation, num_samples=1000)
    image, mask = explanation.get_image_and_mask(label[i], positive_only=False, num_features=7, hide_rest=False)
    image=image.astype(np.uint8)
    plt.imshow(image)
    plt.imsave('{}/fig3_{}.jpg'.format(output_fpath, i), image)
    # plt.show()
    plt.close()
    print('*** Save image {}/fig3_{}.jpg!'.format(output_fpath, i))
