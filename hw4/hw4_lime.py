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

# Handle argv
train_fpath = sys.argv[1]
output_fpath = sys.argv[2]
model_fpath = sys.argv[3]
output_fpath = output_fpath if output_fpath[-1] != '/' else output_fpath[:-1]
print('# Training data : {}'.format(train_fpath))
print('# Output path   : {}'.format(output_fpath))
print('# Model         : {}'.format(model_fpath))

label = [1, 2, 3, 4, 5, 6]
image_ids = [299, 9, 25, 70, 81, 94]

# Load data
def load_train(train_fpath):
    normalization = False
    data = pd.read_csv(train_fpath)
    Y_label = np.array(data['label'].values, dtype=int)
    X_feature = data['feature'].values
    X_train, Y_train = [], []
    print(Y_label)
    for l, id in enumerate(image_ids):
        features = X_feature[id]
        Y_train.append(label[l])
        split_features = [ [int(i),int(i),int(i)] for i in features.split(' ') ]
        matrix_features = np.array(split_features).reshape(48, 48, 3)
        # np.append(matrix_features,np.zeros(48, 48))
        X_train.append(matrix_features)
    if normalization == True:
        X_train = np.array(X_train, dtype=float) / 255.0
    else:
        X_train = np.array(X_train, dtype=float)
    Y_train = np.array(Y_train)
    return X_train, Y_train

# def load_train(train_fpath):
#     normalization = False
#     data = pd.read_csv(train_fpath)
#     Y_train = np.array(data['label'].values, dtype=int)
#     X_train = []
#     for features in data['feature'].values:
#         split_features = [ int(i) for i in features.split(' ') ]
#         matrix_features = np.array(split_features).reshape(48, 48, 1)
#         #matrix_features = np.array(split_features).reshape(48*48)
#         X_train.append(matrix_features)
#         break
#     if normalization == True:
#         X_train = np.array(X_train, dtype=float) / 255.0
#     else:
#         X_train = np.array(X_train, dtype=float)
#     return X_train, Y_train

print('# Loading data...')
X_train, Y_train = load_train(train_fpath)
Y_train = np_utils.to_categorical(Y_train, 7)
print(X_train.shape)
# print(X_train)
print(Y_train)

# Load model
print('# Loading model...')
model = load_model(model_fpath)

x_train_rgb = X_train

# from skimage.segmentation import slic, mark_boundaries
# img=x_train_rgb[0]
# plt.imshow(mark_boundaries(img, slic(img)))
# plt.show()

# from skimage.segmentation import slic, mark_boundaries
# def predict(input):
#     gray = np.array(input[:,:,:,0], ndmin=4)
#     gray.resize(input.shape[0],48,48,1)
#     return model.predict(gray)

# def segmentation(input):
#     return slic(input)

# for i, idx in enumerate(image_ids):
#     explainer = lime_image.LimeImageExplainer()
#     explaination = explainer.explain_instance(
#                                     image=x_train_rgb[i],
#                                     classifier_fn=predict,
#                                     segmentation_fn=segmentation
#                                 )
#     image, mask = explaination.get_image_and_mask(
#                                     label=label[i],
#                                     positive_only=True,
#                                     hide_rest=True,
#                                     num_features=7,
#                                     min_weight=0.0
#                                 )
#     print(type(image))
#     print(image)
#     print(image.shape)
#     image.astype(int)
#     plt.imshow(image)
#     plt.show()
#     plt.imshow(mask)
#     plt.show()
#     plt.imshow(mark_boundaries(image / 2 + 0.5, mask))
#     plt.show()
#     # plt.imsave('{}/fig3_{}.jpg'.format(output_fpath, i), image)
#     print('*** Save image {}/fig3_{}.jpg!'.format(output_fpath, i))
from keras_explain.lime_ribeiro import Lime
from keras_explain.prediction_diff import PredictionDiff
image = X_train[0]
explainer = PredictionDiff(model)
exp_pos, exp_neg = explainer.explain(image, 0)