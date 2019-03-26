import sys
import csv
import pickle
import numpy as np
import pandas as pd

from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils, to_categorical

def load_test(test_fpath):
    data = pd.read_csv(test_fpath)
    id_test = np.array(data['id'].values, dtype=int)
    X_test = []
    for features in data['feature'].values:
        split_features = [ int(i) for i in features.split(' ') ]
        matrix_features = np.array(split_features).reshape(48, 48, 1)
        X_test.append(matrix_features)
    X_test = np.array(X_test)
    return X_test, id_test

test_fpath = sys.argv[1]
output_fpath = sys.argv[2]
model_fpath = sys.argv[3]
print('# Test    : {}'.format(test_fpath))
print('# Outrput : {}'.format(output_fpath))
print('# Model   : {}'.format(model_fpath))

X_test, id_test = load_test(test_fpath)

model = load_model(model_fpath)
prediction = model.predict(X_test)
#print(prediction)

with open(output_fpath, 'w') as f:
    f.write('id,label\n')
    for i, v in enumerate(prediction):
        f.write('%d,%d\n' %(i, np.argmax(v)))

