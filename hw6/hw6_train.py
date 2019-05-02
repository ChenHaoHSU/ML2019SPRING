import sys, os, random
import numpy as np
import pandas as pd
from multiprocessing import Pool

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import BatchNormalization
from keras.layers import GRU, LSTM, SimpleRNN
from keras.utils import np_utils, to_categorical
from keras.models import load_model

import jieba
from gensim.models import Word2Vec

MAX_LENGTH = 40
EMBEDDING_DIM = 100
LOAD_W2V = False
SHUFFLE_SPLIT = False
BATCH_SIZE = 100
EPOCHS = 16
VAL_RATIO = 0.1

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
    print('# [Info] Loading jieba...')
    jieba.load_userdict(dict_fpath)
    P = Pool(processes=4) 
    X_segment = P.map(self.tokenize, X_train)
    P.close()
    P.join()
    return X_segment

def tokenize(sentence):
    tokens = []
    for word in list(jieba.cut(sent, cut_all=False)):
        if word[0] == 'B': continue
        tokens.append(word)
    return tokens

def word_to_vector(X_segment):
    print('# [Info] Building W2V model...')
    if LOAD_W2V == True:
        w2v_model = Word2Vec.load(w2v_fpath)
    else:
        w2v_model = Word2Vec(X_segment, size=EMBEDDING_DIM, window=6, min_count=3, workers=8, iter=25)
        w2v_model.save(w2v_fpath)
    X_train = np.zeros((len(X_segment), MAX_LENGTH, EMBEDDING_DIM))
    for i in range(len(X_segment)):
        print('\r# [Info] Converting texts to vectors... {} / {}'.format(i+1, len(X_segment)), end='', flush=True)
        for j in range(min(len(X_segment[i]), MAX_LENGTH)):
            try:
                vector = w2v_model[X_segment[i][j]]
                X_train[i][j] = (vector - vector.mean(0)) / (vector.std(0) + 1e-20)
            except KeyError as e:
                pass
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
# Y_train = np_utils.to_categorical(Y_train, 2)

''' Validation set '''
X_train, Y_train, X_val, Y_val = split_train_val(X_train, Y_train, VAL_RATIO)
print('# [Info] train / val : {} / {}.'.format(len(X_train), len(X_val)))

model = new_model()
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

''' Train Train Train '''
# train_history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
train_history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(X_val, Y_val))

''' Save model '''
model.save(model_fpath)
print('# [Info] Model saved. \'{}\''.format(model_fpath))
print('Done!')
