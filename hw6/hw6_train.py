import sys, os, random
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dense, Dropout, Activation
from keras.layers import ZeroPadding2D, BatchNormalization
from keras.layers import GRU, LSTM
from keras.optimizers import SGD, Adam
from keras.utils import np_utils, to_categorical
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

import jieba
from gensim.models import word2vec
import emoji

MAX_SEQUENCE_LENGTH = 39
EMBEDDING_DIM = 100
MAX_NB_WORDS = 8192

''' Handle argv '''
# bash hw6_train.sh <train_x file> <train_y file> <test_x.csv file> <dict.txt.big file>
X_train_fpath = sys.argv[1]
Y_train_fpath = sys.argv[2]
X_test_fpath = sys.argv[3]
dict_fpath = sys.argv[4]
w2v_fpath = sys.argv[5]
model_fpath = sys.argv[6]
print('# [Info] Argv')
print('    - X train file : {}'.format(X_train_fpath))
print('    - Y train file : {}'.format(Y_train_fpath))
print('    - X test file  : {}'.format(X_test_fpath))
print('    - Dict file    : {}'.format(dict_fpath))
print('    - W2V file     : {}'.format(w2v_fpath))
print('    = Model file   : {}'.format(model_fpath))

def load_X(fpath):
    data = pd.read_csv(fpath)
    return np.array(data['comment'].values, dtype=str)

def load_Y(fpath):
    data = pd.read_csv(fpath)
    return np.array(data['label'].values, dtype=int)

def split_train_val(X, Y, val_ratio, shuffle=False):
    assert len(X) == len(Y)
    if shuffle == True:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
    train_len = int(len(X) * (1.0 - val_ratio))
    return X[:train_len], Y[:train_len], X[train_len:], Y[train_len:]

''' Fix random seeds '''
random.seed(0)
np.random.seed(0)

''' Load training data '''
print('# [Info] Loading training data...')
X_train = load_X(X_train_fpath)
Y_train = load_Y(Y_train_fpath)
Y_train = np_utils.to_categorical(Y_train, 2)
assert len(X_train) == len(Y_train)
print('# [Info] {} training data loaded.'.format(len(X_train)))

''' Load dict.txt '''
print('# [Info] Loading txt dict...')
jieba.load_userdict(dict_fpath)
X_train_list = [ list(jieba.cut(sent, cut_all=False)) for sent in X_train ]

''' word2vec '''
print('# [Info] W2V model.')
w2v_model = word2vec.Word2Vec(size=EMBEDDING_DIM, window=4, min_count=1, workers=4)
w2v_model.save(w2v_fpath)

print('Converting texts to vectors...')
X_train = np.zeros((len(X_train_list), MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))
for n in range(len(X_train_list)):
    for i in range(min(len(X_train_list[n]), MAX_SEQUENCE_LENGTH)):
        try:
            print ('Word', X_train_list[n][i], 'is in dictionary.')
            vector = w2v_model[X_train_list[n][i]]
            X_train[n][i] = (vector - vector.mean(0)) / (vector.std(0) + 1e-20)
        except KeyError as e:
            # print ('Word', X_train_list[n][i], 'is not in dictionary.')

''' Split validation set '''
print('# [Info] Splitting training data into train and val set...')
val_ratio = 0.01
X_train, Y_train, X_val, Y_val = split_train_val(X_train, Y_train, val_ratio)
assert len(X_train) == len(Y_train) and len(X_val) == len(Y_val)
print('# [Info] train / val : {} / {}.'.format(len(X_train), len(X_val)))

print('Shape of X_train tensor:', X_train.shape)
print('Shape of Y_train tensor:', Y_train.shape)

''' Build model '''
dropout = 0.25
print('# [Info] Building model...')
model = Sequential()
model.add(GRU(units=128, input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM), dropout=0.1, recurrent_dropout=0.1))
model.add(Dense(units=256, kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

''' Print model summary '''
model.summary()

''' Compile model '''
print('# [Info] Compling model...')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

''' Train Train Train '''
BATCH_SIZE = 128
EPOCHS = 16
# train_history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
train_history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(X_val, Y_val))

''' Save model '''
model.save(model_fpath)
print('# [Info] Model saved. \'{}\''.format(model_fpath))
print('Done!')
