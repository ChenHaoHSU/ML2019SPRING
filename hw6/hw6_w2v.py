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

MAX_LENGTH = 40
EMBEDDING_DIM = 200
WINDOW = 5
MIN_COUNT = 3
WORKERS = 8
ITER = 30

''' Handle argv '''
X_train_fpath = sys.argv[1]
Y_train_fpath = sys.argv[2]
X_test_fpath = sys.argv[3]
dict_fpath = sys.argv[4]
w2v_fpath = sys.argv[5]
print('# [Info] Argv')
print('    - X train file : {}'.format(X_train_fpath))
print('    - Y train file : {}'.format(Y_train_fpath))
print('    - X test file  : {}'.format(X_test_fpath))
print('    - Dict file    : {}'.format(dict_fpath))
print('    = W2V file     : {}'.format(w2v_fpath))

''' Fix random seeds '''
random.seed(0)
np.random.seed(0)

def load_X(fpath):
    data = pd.read_csv(fpath)
    return np.array(data['comment'].values, dtype=str)

def load_Y(fpath):
    data = pd.read_csv(fpath)
    return np.array(data['label'].values, dtype=int)

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
    print('# [Info] Building w2v model...')
    print('#    - Total data: {}'.format(len(X)))
    embed = Word2Vec(X, size=EMBEDDING_DIM, window=WINDOW, min_count=MIN_COUNT, workers=WORKERS, iter=ITER)
    print('#    - Vocab size: {}'.format(len(embed.wv.vocab)))
    print('#    - Model saved: {}'.format(w2v_fpath))
    embed.save(w2v_fpath)
    return embed

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
