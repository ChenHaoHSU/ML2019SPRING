import sys
import csv
import math
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
from keras.preprocessing.image import ImageDataGenerator

train_fpath = sys.argv[1]
model_fpath = sys.argv[2]
print('# Train : {}'.format(train_fpath))
print('# Model : {}'.format(model_fpath))

def load_train(train_fpath):
    data = pd.read_csv(train_fpath)
    Y_train = np.array(data['label'].values, dtype=int)
    X_train = []
    for features in data['feature'].values:
        split_features = [ int(i) for i in features.split(' ') ]
        matrix_features = np.array(split_features).reshape(48, 48, 1)
        X_train.append(matrix_features)
    X_train = np.array(X_train)
    return X_train, Y_train

# Loading training data
print('# Loading data...')
X_train, Y_train = load_train(train_fpath)
Y_train = np_utils.to_categorical(Y_train, 7)
print('X_train.shape:', X_train.shape)
print('Y_train.shape:', Y_train.shape)

# ImageDataGenerator
datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=True,
	rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.3,
    fill_mode='nearest')
datagen.fit(X_train)

print('# Setting model...')
dropout = 0.25
model = Sequential()
# CNN
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)))
model.add(ZeroPadding2D(padding=(2, 2), data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

for i in range(3):
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

model.add(Flatten())

# DNN
dnn_neurons = [512, 256, 128]
for neurons in dnn_neurons:
    model.add(Dense(neurons, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

model.summary()

print('# Compling model...')
batch_size = 100
epochs = 100
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

print('# Start training...')
# model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)
train_history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True),
                                    epochs=epochs, steps_per_epoch=math.ceil(len(X_train)/batch_size))

train_acc = model.evaluate(X_train, Y_train)
print('\nTrain Acc:', train_acc)

print('# Saving model...')
model.save(model_fpath)
