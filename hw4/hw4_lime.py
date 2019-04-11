import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import np_utils
import keras.backend as K
from lime import lime_image

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

# Load data
def load_train(train_fpath):
    normalization = False
    data = pd.read_csv(train_fpath)
    Y_train = np.array(data['label'].values, dtype=int)
    X_train = []
    i = 0
    for features in data['feature'].values:
        if i in image_ids:
            split_features = [ [int(i),0,0] for i in features.split(' ') ]
            matrix_features = np.array(split_features).reshape(48, 48, 3)
            # np.append(matrix_features,np.zeros(48, 48))
            X_train.append(matrix_features)
        i += 1
    if normalization == True:
        X_train = np.array(X_train, dtype=float) / 255.0
    else:
        X_train = np.array(X_train, dtype=float)
    return X_train, Y_train

print('# Loading data...')
X_train, Y_train = load_train(train_fpath)
Y_train = np_utils.to_categorical(Y_train, 7)
print(X_train.shape)

# Load model
print('# Loading model...')
model = load_model(model_fpath)

x_train_rgb = X_train

def predict(input):
    gray = input[:,:,:,0]
    print(gray.shape)
    gray.reshape(10,48,48,1)
    return model.predict(gray)

# def segmentation(input):
#     skimage.segmentation.slic()

for i, idx in enumerate(image_ids):
    explainer = lime_image.LimeImageExplainer()
    explaination = explainer.explain_instance(
                                    image=x_train_rgb[i],
                                    classifier_fn=predict,
                                    segmentation_fn=None
                                )
    image, mask = explaination.get_image_and_mask(
                                    label=label[idx],
                                    positive_only=False,
                                    hide_rest=False,
                                    num_features=5,
                                    min_weight=0.0
                                )
    plt.imsave('{}/fig3_{}.jpg'.format(output_fpath, i))
    print('*** Save image {}/fig3_{}.jpg!'.format(output_fpath, i))
