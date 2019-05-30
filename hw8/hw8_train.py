import sys
import csv
import math
import pickle
import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dense, Dropout, Activation
from keras.layers import ZeroPadding2D, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils, to_categorical
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

from keras.applications import mobilenet
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras.models import Model, load_model
from keras import applications
from keras import optimizers
from keras import backend as K

def load_train(train_fpath):
    normalization = False
    data = pd.read_csv(train_fpath)
    Y_train = np.array(data['label'].values, dtype=int)
    
    X_train = np.load('X_train.npy')
    return X_train, Y_train
    
    X_train = []
    for features in data['feature'].values:
        split_features = [ int(i) for i in features.split(' ') ]
        matrix_features = np.array(split_features).reshape(48, 48, 1)
        #matrix_features = np.array(split_features).reshape(48*48)
        X_train.append(matrix_features)
    X_train = np.array(X_train, dtype=float)
    X_train = X_train / 255.0
    np.save('X_train.npy', X_train)
    return X_train, Y_train

def train_val_split(X_train, Y_train, val_size=0.1):
    train_len = int(round(len(X_train)*(1-val_size)))
    return X_train[0:train_len], Y_train[0:train_len], X_train[train_len:None], Y_train[train_len:None]

def build_mobilenet():
    model = MobileNet(input_shape=(48, 48, 1), weights=None, dropout=0.2, classes=7)
    model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])
    return model

def build_cnn():
    dropout = 0.25
    model = Sequential()
    # CNN
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_last', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(BatchNormalization())
    for i in range(2):
        for j in range(2):
            model.add(Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_last'))
            model.add(BatchNormalization())
        for j in range(1):
            model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
            model.add(Dropout(dropout))
    # flatten
    model.add(Flatten())
    # DNN
    dnn_neurons = [512, 256, 128]
    for neurons in dnn_neurons:
        model.add(Dense(neurons, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
    model.add(Dense(units=7, activation='softmax'))
    return model

# Agrv handling
train_fpath = sys.argv[1]
model_fpath = sys.argv[2]
weights_fpath = sys.argv[3]
print('# Train   : {}'.format(train_fpath))
print('# Model   : {}'.format(model_fpath))
print('# Weights : {}'.format(weights_fpath))

# Loading training data
print('# Loading data...')
X_train, Y_train = load_train(train_fpath)
Y_train = np_utils.to_categorical(Y_train, 7)
print('X_train.shape:', X_train.shape)
print('Y_train.shape:', Y_train.shape)

# Split into training set and validation set
val_size = 0.1
X_train, Y_train, X_val, Y_val = train_val_split(X_train, Y_train, val_size)
print('X_train.shape:', X_train.shape)
print('Y_train.shape:', Y_train.shape)
print('X_val.shape:', X_val.shape)
print('Y_val.shape:', Y_val.shape)

# Image augmentation
datagen = ImageDataGenerator(
    featurewise_center=False, featurewise_std_normalization=False,
    width_shift_range=0.2, height_shift_range=0.2,
    horizontal_flip=False, vertical_flip=False,
    rotation_range=12, zoom_range=0.5,
    fill_mode='nearest')
datagen.fit(X_train)

model = build_mobilenet()
model.summary()

print('# Compling model...')
batch_size = 100
epochs = 100
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

print('# Start training...')
# train_history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, Y_val))
# train_history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True),
#                                    epochs=epochs, steps_per_epoch=5*math.ceil(len(X_train)/batch_size))
train_history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True),
                                     steps_per_epoch=5*math.ceil(len(X_train)/batch_size),
                                     validation_data=(X_val, Y_val),
                                     validation_steps=len(X_val)/batch_size,
                                     epochs=epochs)

result = model.evaluate(X_train, Y_train)
print('\nTrain Acc:', result[1])

print('# Saving model...')
model.save(model_fpath)
print('# Saving weights...')
mode.save_weights(weights_fpath)
