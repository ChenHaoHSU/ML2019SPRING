import sys
import csv
import pickle
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils, to_categorical

def load_train(train_fpath):
    data = pd.read_csv(train_fpath)
    Y_train = np.array(data['label'].values, dtype=int)
    X_train = []
    for features in data['feature'].values:
        split_features = [ int(i) for i in features.split(' ') ]
        matrix_features = np.array(split_features).reshape(48, 48)
        X_train.append(matrix_features)
    X_train = np.array(X_train)
    return X_train, Y_train

train_fpath = sys.argv[1]
model_fpath = sys.argv[2]
print('# Train : {}'.format(train_fpath))
print('# Model : {}'.format(model_fpath))

X_train, Y_train = load_train(train_fpath)

# model = Sequential()
# model.add(Dense(input_dim=X_train.shape[1], units=500, activation='relu'))
# for i in range(10):
#     model.add(Dense(units=500, activation='relu'))
# model.add(Dense(units=2, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
# model.fit(X_train, Y_train, batch_size=100, epochs=50)
# result = model.evaluate(X_train, Y_train)
# print('\nTrain Acc:', result[1])

# prediction = model.predict(X_test)

# with open(output_fpath, 'w') as f:
#     f.write('id,label\n')
#     for i, v in enumerate(prediction):
#         f.write('%d,%d\n' %(i+1, np.argmax(v)))
