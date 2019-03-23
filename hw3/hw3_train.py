import sys
import csv
import pickle
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dense, Dropout, Activation
from keras.layers import ZeroPadding2D, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils, to_categorical
from keras.layers.pooling import MaxPooling2D, AveragePooling2D

def load_train(train_fpath):
    data = pd.read_csv(train_fpath)
    Y_train = np.array(data['label'].values, dtype=int)
    X_train = []
    for features in data['feature'].values:
        split_features = [ int(i) for i in features.split(' ') ]
        matrix_features = [ np.array(split_features).reshape(48, 48) ]
        X_train.append(matrix_features)
    X_train = np.array(X_train)
    return X_train, Y_train

train_fpath = sys.argv[1]
model_fpath = sys.argv[2]
print('# Train : {}'.format(train_fpath))
print('# Model : {}'.format(model_fpath))

X_train, Y_train = load_train(train_fpath)
print(X_train.shape)

model = Sequential()
# CNN
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=(1, 48, 48)))
model.add(MaxPooling2D(pool_size=(2, 2)))
for i in range(2):
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
model.add(Flatten())

# DNN
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(units=7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=100, epochs=50)
result = model.evaluate(X_train, Y_train)
print('\nTrain Acc:', result[1])

model.save(model_fpath)
