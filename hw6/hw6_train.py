import sys, os, random
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import BatchNormalization
from keras.layers import GRU, LSTM, Bidirectional, Embedding
from keras.utils import np_utils, to_categorical

import jieba
from gensim.models import Word2Vec
import emoji

MAX_LENGTH = 40
EMBEDDING_DIM = 200
WINDOW = 5
MIN_COUNT = 3
WORKERS = 8
ITER = 30

LOAD_W2V = True

VAL_RATIO = 0.1
BATCH_SIZE = 100
EPOCHS = 12

DROPOUT = 0.3
RECURRENT_DROPOUT = 0.3

''' Handle argv '''
# bash hw6_train.sh <train_x file> <train_y file> <test_x.csv file> <dict.txt.big file>
X_train_fpath = sys.argv[1]
Y_train_fpath = sys.argv[2]
X_test_fpath = sys.argv[3]
dict_fpath = sys.argv[4]
w2v_fpath = sys.argv[5]
model_fpath = sys.argv[6]
randseed = int(sys.argv[7])
print('# [Info] Argv')
print('    - X train file : {}'.format(X_train_fpath))
print('    - Y train file : {}'.format(Y_train_fpath))
print('    - X test file  : {}'.format(X_test_fpath))
print('    - Dict file    : {}'.format(dict_fpath))
print('    - W2V file     : {}'.format(w2v_fpath))
print('    = Model file   : {}'.format(model_fpath))
print('    - seed         : {}'.format(randseed))

''' Fix random seeds '''
random.seed(randseed)
np.random.seed(randseed)

def load_X(fpath):
    data = pd.read_csv(fpath)
    return np.array(data['comment'].values, dtype=str)

def load_Y(fpath):
    data = pd.read_csv(fpath)
    return np.array(data['label'].values, dtype=int)

def split_train_val(X, Y, val_ratio, shuffle=True):
    assert X.shape[0] == Y.shape[0]
    if shuffle == True:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
    train_len = int(X.shape[0] * (1.0 - val_ratio))
    return X[:train_len], Y[:train_len], X[train_len:], Y[train_len:]

def text_segmentation(X):
    segment = []
    for i, sentence in enumerate(X):
        print('\r# [Info] Segmenting sentences... {} / {}'.format(i+1, len(X)), end='', flush=True)
        word_list = []
        for word in list(jieba.cut(sentence, cut_all=False)):
            if word[0] == 'B': continue
            word_list.append(word)
        segment.append(word_list)
    print('', flush=True)
    return segment

def build_embed(X):
    if LOAD_W2V == True:
        print('# [Info] Loading w2v model...')
        embed = Word2Vec.load(w2v_fpath)
    else:
        print('# [Info] Building w2v model...')
        print('#    - Total data: {}'.format(len(X)))
        embed = Word2Vec(X, size=EMBEDDING_DIM, window=WINDOW, min_count=MIN_COUNT, workers=WORKERS, iter=ITER)
        print('#    - Vocab size: {}'.format(len(embed.wv.vocab)))
        print('#    - Model saved: {}'.format(w2v_fpath))
        embed.save(w2v_fpath)
    return embed

def word_to_vector(embed, segment):
    vectors = np.zeros((len(segment), MAX_LENGTH, EMBEDDING_DIM))
    for i in range(len(segment)):
        print('\r# [Info] Converting texts to vectors... {} / {}'.format(i+1, len(segment)), end='', flush=True)
        for j in range(min(len(segment[i]), MAX_LENGTH)):
            try:
                vector = embed[segment[i][j]]
                vectors[i][j] = (vector - vector.mean(0)) / (vector.std(0) + 1e-20)
            except KeyError as e:
                pass
    print('', flush=True)
    return vectors

def build_model():
    print('# [Info] Building model...')
    model = Sequential()
    model.add(Bidirectional(GRU(256, dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT,
                  return_sequences=True, activation='tanh'),
                  input_shape=(MAX_LENGTH, EMBEDDING_DIM)))
    model.add(Bidirectional(GRU(256, dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT,
                  return_sequences=True, activation='tanh')))
    model.add(Bidirectional(GRU(256, dropout=DROPOUT, recurrent_dropout=RECURRENT_DROPOUT,
                  return_sequences=False, activation='tanh')))
    neurons = [256]
    for neuron in neurons:
        model.add(Dense(neuron, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(DROPOUT))
    model.add(Dense(2, activation='sigmoid'))
    return model

''' Load training data '''
X_train = load_X(X_train_fpath)
Y_train = load_Y(Y_train_fpath)
X_test = load_X(X_test_fpath)
assert X_train.shape[0] == Y_train.shape[0]
print('# [Info] {} training data loaded.'.format(len(X_train)))

''' Preprocess '''
print('# [Info] Loading JIEBA...')
jieba.load_userdict(dict_fpath)
X_train = X_train[0:119018]
Y_train = Y_train[0:119018]
X_train_segment = text_segmentation(X_train)
X_test_segment = text_segmentation(X_test)
embed = build_embed(X_train_segment + X_test_segment)
X_train = word_to_vector(embed, X_train_segment)
Y_train = np_utils.to_categorical(Y_train, 2)

''' Validation set '''
X_train, Y_train, X_val, Y_val = split_train_val(X_train, Y_train, VAL_RATIO)
print('# [Info] train / val : {} / {}.'.format(len(X_train), len(X_val)))

''' Build model '''
model = build_model()
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

''' Train Train Train '''
train_history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(X_val, Y_val))
result = model.evaluate(X_train, Y_train)
print('# [Info] Train Acc:', result[1])
result = model.evaluate(X_val, Y_val)
print('# [Info] Val Acc:', result[1])

''' Save model '''
model.save(model_fpath)
print('# [Info] Model saved: {}'.format(model_fpath))
print('Done!')
