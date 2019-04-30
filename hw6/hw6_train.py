import sys, os, random
import numpy as np
import pandas as pd

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dense, Dropout, Activation
from keras.layers import ZeroPadding2D, BatchNormalization
from keras.layers import GRU, LSTM, Input, Embedding
from keras.layers import Input
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

''' Fix random seeds '''
random.seed(0)
np.random.seed(0)

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

def segment(X):
    jieba.load_userdict(dict_fpath)
    X_seg = []
    filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n '+'～＠＃＄％︿＆＊（）！？⋯  ，。'
    for sent in X:
        tmp_list = []
        for c in filters:
            sent = sent.replace(c, '')
        for word in list(jieba.cut(sent, cut_all=False)):
            if word[0] == 'B': continue
            tmp_list.append(word)
        X_seg.append(tmp_list)
    return X_seg

def w2v(X_seg):
    print('# [Info] W2V model.')
    w2v_model = word2vec.Word2Vec(size=EMBEDDING_DIM, window=5, min_count=3, workers=8, iter=20)
    w2v_model.save(w2v_fpath)
    print(w2v_model)
    print(w2v_model.wv.vocab)
    print('Converting texts to vectors...')
    X_train = np.zeros((len(X_train_list), MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))
    for n in range(len(X_train_list)):
        for i in range(min(len(X_train_list[n]), MAX_SEQUENCE_LENGTH)):
            try:
                # print ('Word', X_train_list[n][i], 'is in dictionary.')
                vector = w2v_model[X_train_list[n][i]]
                X_train[n][i] = vector
                #X_train[n][i] = (vector - vector.mean(0)) / (vector.std(0) + 1e-20)
            except KeyError as e:
                pass
                # print ('Word', X_train_list[n][i], 'is not in dictionary.')

dropout = 0.25
def new_model():
    model = Sequential()
    model.add(embedding_layer)
    model.add(GRU(16))    
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model

def new_model2():
    print('# [Info] Building model...')
    model = Sequential()
    model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2, input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM), 
                    return_sequences=True, activation='tanh'))
    model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2,
                    return_sequences=True, activation='tanh'))
    model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2,
                    return_sequences=False, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(2, activation='softmax'))
    return model

''' Load training data '''
print('# [Info] Loading training data...')
X_train = load_X(X_train_fpath)
Y_train = load_Y(Y_train_fpath)
Y_train = np_utils.to_categorical(Y_train, 2)
assert len(X_train) == len(Y_train)
print('# [Info] {} training data loaded.'.format(len(X_train)))

''' Load dict.txt '''
print('# [Info] Loading txt dict...')
X_seg = segment(X_train)
X_train = w2v(X_seg)
print('# [Info] Splitting training data into train and val set...')
val_ratio = 0.01
X_train, Y_train, X_val, Y_val = split_train_val(X_train, Y_train, val_ratio)
assert len(X_train) == len(Y_train) and len(X_val) == len(Y_val)
print('# [Info] train / val : {} / {}.'.format(len(X_train), len(X_val)))

model = new_model()
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

''' Train Train Train '''
BATCH_SIZE = 100
EPOCHS = 16
# train_history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
train_history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(X_val, Y_val))

''' Save model '''
model.save(model_fpath)
print('# [Info] Model saved. \'{}\''.format(model_fpath))
print('Done!')
