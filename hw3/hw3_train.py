# import sys
# import csv
# import pickle
# import numpy as np
# import pandas as pd

# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Flatten
# from keras.layers import Dense, Dropout, Activation
# from keras.layers import ZeroPadding2D, BatchNormalization
# from keras.optimizers import SGD, Adam
# from keras.utils import np_utils, to_categorical
# from keras.layers.pooling import MaxPooling2D, AveragePooling2D

# def load_train(train_fpath):
#     data = pd.read_csv(train_fpath)
#     Y_train = np.array(data['label'].values, dtype=int)
#     X_train = []
#     for features in data['feature'].values:
#         split_features = [ int(i) for i in features.split(' ') ]
#         matrix_features = np.array(split_features).reshape(48, 48, 1)
#         X_train.append(matrix_features)
#     X_train = np.array(X_train)
#     return X_train, Y_train

# train_fpath = sys.argv[1]
# model_fpath = sys.argv[2]
# print('# Train : {}'.format(train_fpath))
# print('# Model : {}'.format(model_fpath))

# X_train, Y_train = load_train(train_fpath)
# Y_train = np_utils.to_categorical(Y_train, 7)
# print(Y_train.shape)

# model = Sequential()
# # CNN
# model.add(Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)))
# model.add(ZeroPadding2D(padding=(2, 2), data_format='channels_last'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))

# for i in range(2):
#     model.add(Conv2D(128, (3, 3), activation='relu'))
#     model.add(BatchNormalization())
#     model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
#     model.add(Conv2D(128, (3, 3), activation='relu'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))


# # DNN
# model.add(Dense(512, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.25))
# model.add(Dense(512, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.25))
# model.add(Dense(512, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.25))
# model.add(Dense(units=7, activation='softmax'))

# model.summary()

# model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
# model.fit(X_train, Y_train, batch_size=100, epochs=50)
# result = model.evaluate(X_train, Y_train)
# print('\nTrain Acc:', result[1])

# model.save(model_fpath)
# sys.exit()

##################################################
##################################################
##################################################
##################################################

import sys
import csv
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Activation, ZeroPadding2D
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils import np_utils

#load data from train.csv
train_data = sys.argv[1]
mytrain = pd.read_csv(train_data)

train_x = []
val_x = []
train_y = []
val_y = []

for i in range(len(mytrain)):
    if(i%20 == 7):
        val_x.append(np.array(mytrain.iloc[i,1].split()).reshape(48,48,1).astype(float)/256)
    else:
        temp = np.array(mytrain.iloc[i,1].split()).reshape(48,48,1).astype(float)
        temp /= 256
        train_x.append(temp)
        train_x.append(np.flip(temp,axis=1))
train_x = np.array(train_x)
val_x = np.array(val_x)

for i in range(len(mytrain)):
    if(i%20 == 7):
        val_y.append(int(mytrain.iloc[i,0]))
    else:
        train_y.append(int(mytrain.iloc[i,0]))
        train_y.append(int(mytrain.iloc[i,0]))
train_y = np.array(train_y)
train_y = np_utils.to_categorical(train_y, 7)
val_y = np.array(val_y)
val_y = np_utils.to_categorical(val_y, 7)

#parameters
epochs = 90

#ImageGenerator
datagen_train = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.5,
	rotation_range=13,
    horizontal_flip=False,
    fill_mode='nearest')

# datagen_train.fit(train_x)

print(train_y.shape)
sys.exit()

#CNN
model = Sequential()
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)))
model.add(ZeroPadding2D(padding=(2, 2), data_format='channels_last'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

for i in range(2):
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

model.add(Flatten())

#DNN:
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(units=7, activation='softmax'))

#optimizers
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

#compile
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

#earlystopping
filepath="weights_early_1.hdf5"
checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
checkpoint2 = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
callbacks_list = [checkpoint1,checkpoint2]

#print
model.summary()

#fit model
batch_sz = 200
train_history = model.fit_generator(datagen_train.flow(train_x,train_y, batch_size=batch_sz,shuffle=True),
                    steps_per_epoch=3*(math.floor(len(train_x)/batch_sz)), epochs=epochs,
                    validation_data=(val_x, val_y),
                    validation_steps=len(val_x)/batch_sz,
                    callbacks=callbacks_list, verbose=1)

#show_train_history(train_history, 'acc', 'val_acc')

#save model
model.save('model_5.hdf5')

