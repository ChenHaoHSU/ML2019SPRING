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

TEST = True
NUM = 119018
VAL_RATIO = 0.1
BATCH_SIZE = 100
EPOCHS = 20
DROPOUT = 0.2

LOAD_W2V = True
MAX_LENGTH = 40
EMBEDDING_DIM = 200
WINDOW = 5
MIN_COUNT = 20
WORKERS = 8
ITER = 30

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
        for word in jieba.cut(sentence, cut_all=False):
            if word[0] == 'B': continue
            word_list.append(word)
        segment.append(word_list)
    print('', flush=True)
    return segment

def build_embed(X):
    if LOAD_W2V == True:
        print('# [Info] Loading w2v model...')
        embed = Word2Vec.load(w2v_fpath)
        print('#    - Vocab size: {}'.format(len(embed.wv.vocab)))
    else:
        print('# [Info] Building w2v model...')
        print('#    - Total data: {}'.format(len(X)))
        embed = Word2Vec(X, size=EMBEDDING_DIM, window=WINDOW, min_count=MIN_COUNT, workers=WORKERS, iter=ITER)
        print('#    - Vocab size: {}'.format(len(embed.wv.vocab)))
        print('#    - Model saved: {}'.format(w2v_fpath))
        embed.save(w2v_fpath)
    return embed

def build_dict(embed):
    word_dict = dict()
    cnt = 0
    for i, word in enumerate(embed.wv.vocab):
        print('\r# [Info] Building word dictionary... {} / {}'.format(i+1, len(embed.wv.vocab)), end='', flush=True)
        word_dict[word] = cnt
        cnt += 1
    print('', flush=True)
    return word_dict

def sentence_to_bag(word_dict, segment):
    vectors = np.zeros((len(segment), len(word_dict)), dtype=int)
    for i, sentence in enumerate(segment):
        print('\r# [Info] Sentence to bag... {} / {}'.format(i+1, len(segment)), end='', flush=True)
        for j, word in enumerate(sentence):
            if word in word_dict:
                vectors[i, word_dict[word]] += 1
    print('', flush=True)
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
    X_test_segment = text_segmentation(X_test)
    embed = build_embed(X_test_segment)
    word_dict = build_dict(embed)
    X_test = sentence_to_bag(word_dict, X_test_segment)
    model = load_model(model_fpath)
    prediction = model.predict(X_test, batch_size=BATCH_SIZE)
    with open('bow0.csv', 'w') as f:
        f.write('id,label\n')
        for i, v in enumerate(prediction):
            f.write('%d,%d\n' %(i, np.argmax(v)))
    sys.exit()
else:
    print('# [Info] Loading JIEBA...')
    jieba.load_userdict(dict_fpath)
    X_train = X_train[0:NUM]
    Y_train = Y_train[0:NUM]
    X_train_segment = text_segmentation(X_train)
    embed = build_embed(X_train_segment)
    word_dict = build_dict(embed)
    X_train = sentence_to_bag(word_dict, X_train_segment)
    Y_train = np_utils.to_categorical(Y_train, 2)

''' Split validation set '''
X_train, Y_train, X_val, Y_val = split_train_val(X_train, Y_train, VAL_RATIO)
print('# [Info] train / val : {} / {}.'.format(len(X_train), len(X_val)))

''' Build model '''
def build_model(size):
    print('# [Info] Building model...')
    model = Sequential()
    model.add(Dense(units=128, input_dim=size, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(DROPOUT))
    neurons = [128, 128]
    for neuron in neurons:
        model.add(Dense(neuron, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(DROPOUT))
    model.add(Dense(2, activation='sigmoid'))
    return model

''' Build model '''
model = build_model(len(word_dict))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

''' Train Train Train '''
train_history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(X_val, Y_val))
result = model.evaluate(X_train, Y_train, batch_size=BATCH_SIZE)
print('# [Info] Train Acc:', result[1])

def dump_train_history(train_history):
    model_type = 'BOW'
    item_type = ['acc', 'val_acc', 'loss', 'val_loss' ]
    for item in item_type:
        filename = '{}_{}.csv'.format(model_type, item)
        data = train_history.history[item]
        with open(filename, 'w') as f:
            for i in enumerate(data):
                f.write('{}\n'.format(i[1]))
dump_train_history(train_history)

''' Save model '''
model.save(model_fpath)
print('# [Info] Model saved: {}'.format(model_fpath))
print('Done!')
