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

MAX_LENGTH = 40
EMBEDDING_DIM = 100

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
    assert X.shape[0] == Y.shape[0]
    if shuffle == True:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
    train_len = int(X.shape[0] * (1.0 - val_ratio))
    return X[:train_len], Y[:train_len], X[train_len:], Y[train_len:]

def text_segmentation(X_train):
    print('# [Info] Segmenting text...')
    jieba.load_userdict(dict_fpath)
    X_segment = []
    filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n '+'～＠＃＄％︿＆＊（）！？⋯  ，。'
    filters = ''
    for i, sent in enumerate(X_train):
        print('\r#   - Segmenting ({} / {})'.format(i+1, len(X_train)), end='', flush=True)
        tmp_list = []
        for c in filters:
            sent = sent.replace(c, '')
        for word in list(jieba.cut(sent, cut_all=False)):
            if word[0] == 'B': continue
            tmp_list.append(word)
        X_segment.append(tmp_list)
    print('', flush=True)
    return X_segment

def word_to_vector(X_segment):
    print('# [Info] Building W2V model...')
    w2v_model = word2vec.Word2Vec(X_segment, size=EMBEDDING_DIM, window=6, min_count=3, workers=8, iter=25)
    w2v_model.save(w2v_fpath)
    X_train = np.zeros((len(X_segment), MAX_LENGTH, EMBEDDING_DIM))
    for i in range(len(X_segment)):
        print('\r#   - Converting texts to vectors ({} / {})'.format(i+1, len(X_segment)), end='', flush=True)
        for j in range(min(len(X_segment[i]), MAX_LENGTH)):
            try:
                vector = w2v_model[X_segment[i][j]]
                # X_train[i][j] = vector
                X_train[i][j] = (vector - vector.mean(0)) / (vector.std(0) + 1e-20)
            except KeyError as e:
                pass
                # print ('Word', X_segment[n][i], 'is not in dictionary.')
    print('', flush=True)
    return X_train

def new_model():
    print('# [Info] Building model...')
    DROPOUT = 0.2
    model = Sequential()
    model.add(GRU(256, dropout=0.2, recurrent_dropout=0.2,
                  return_sequences=True, activation='sigmoid',
                  input_shape=(MAX_LENGTH, EMBEDDING_DIM)))
    model.add(GRU(256, dropout=0.2, recurrent_dropout=0.2,
                  return_sequences=True, activation='sigmoid'))
    model.add(GRU(256, dropout=0.2, recurrent_dropout=0.2,
                  return_sequences=False, activation='sigmoid'))
    neurons = [512, 256, 128]
    for neuron in neurons:
        model.add(Dense(neuron, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(DROPOUT))
    model.add(Dense(2, activation='softmax'))
    return model

''' Load training data '''
X_train = load_X(X_train_fpath)
Y_train = load_Y(Y_train_fpath)
assert X_train.shape[0] == Y_train.shape[0]
print('# [Info] {} training data loaded.'.format(len(X_train)))

''' Preprocess '''
X_segment = text_segmentation(X_train)
X_train = word_to_vector(X_segment)

Y_train = np_utils.to_categorical(Y_train, 2)

''' Validation set '''
VAL_RATIO = 0.1
X_train, Y_train, X_val, Y_val = split_train_val(X_train, Y_train, VAL_RATIO)
print('# [Info] train / val : {} / {}.'.format(len(X_train), len(X_val)))

model = new_model()
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

''' Train Train Train '''
BATCH_SIZE = 100
EPOCHS = 16
# train_history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
train_history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(X_val, Y_val))

result = model.evaluate(X_train, Y_train)
print('\nTrain Acc:', result[1])

''' Save model '''
model.save(model_fpath)
print('# [Info] Model saved. \'{}\''.format(model_fpath))
print('Done!')
