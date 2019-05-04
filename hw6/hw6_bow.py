import sys, os, random
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import BatchNormalization
from keras.utils import np_utils, to_categorical

from keras.models import load_model

import jieba
from gensim.models import Word2Vec

TEST = False

VAL_RATIO = 0.1
BATCH_SIZE = 100
EPOCHS = 12

DROPOUT = 0.2

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

def build_dict(X):
    word_dict = dict()
    cnt = 0
    for i, sentence in enumerate(X):
        print('\r# [Info] Building word dictionary... {} / {}'.format(i+1, len(X)), end='', flush=True)
        for j, word in enumerate(sentence):
            if word[0] == 'B': continue
            if word in [' ', '  ', '']: continue
            if word not in word_dict:
                word_dict[word] = cnt
                cnt += 1
    print('', flush=True)
    return word_dict

def sentence_to_bag(word_dict, segment):
    vectors = np.zeros((len(segment), len(word_dict)), dtype=int)
    for i, sentence in enumerate(segment):
        for j, word in enumerate(sentence):
            if word in word_dict:
                vectors[i, word_dict[word]] += 1
    print(vectors.shape)
    return vectors

''' Load training data '''
X_train = load_X(X_train_fpath)
Y_train = load_Y(Y_train_fpath)
X_test = load_X(X_test_fpath)
assert X_train.shape[0] == Y_train.shape[0]
print('# [Info] {} training data loaded.'.format(len(X_train)))

''' Preprocess '''
if TEST == True:
    print('# [Info] Loading JIEBA...')
    jieba.load_userdict(dict_fpath)
    X_train = X_train[0:119018]
    X_train_segment = text_segmentation(X_train)
    X_test_segment = text_segmentation(X_test)
    word_dict = build_dict(X_train_segment)
    X_test = sentence_to_bag(word_dict, X_train_segment):
    model = load_model(model_fpath)
    prediction = model.predict(X_test)
    with open('output.csv', 'w') as f:
        f.write('id,label\n')
        for i, v in enumerate(prediction):
            f.write('%d,%d\n' %(i, np.argmax(v)))
    sys.exit()
else:
    print('# [Info] Loading JIEBA...')
    jieba.load_userdict(dict_fpath)
    X_train = X_train[0:119018]
    Y_train = Y_train[0:119018]
    X_train_segment = text_segmentation(X_train)
    X_test_segment = text_segmentation(X_test)
    word_dict = build_dict(X_train_segment)
    X_train = sentence_to_bag(word_dict, X_train_segment):
    Y_train = np_utils.to_categorical(Y_train, 2)

''' Split validation set '''
X_train, Y_train, X_val, Y_val = split_train_val(X_train, Y_train, VAL_RATIO)
print('# [Info] train / val : {} / {}.'.format(len(X_train), len(X_val)))

''' Build model '''
def build_model():
    print('# [Info] Building model...')
    model = Sequential()
    model.add(Dense(units=512, input_dim=len(word_dict), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    neurons = [256, 128]
    for neuron in neurons:
        model.add(Dense(neuron, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
    model.add(Dense(2, activation='softmax'))
    return model

''' Build model '''
model = build_model()
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

''' Train Train Train '''
train_history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(X_val, Y_val))
result = model.evaluate(X_train, Y_train)
print('# [Info] Train Acc:', result[1])

''' Save model '''
model.save(model_fpath)
print('# [Info] Model saved: {}'.format(model_fpath))
print('Done!')