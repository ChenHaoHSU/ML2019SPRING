import sys, os, random
import numpy as np
import pandas as pd
from keras.models import load_model
import jieba
from gensim.models import word2vec

# bash hw6_test.sh <test_x file> <dict.txt.big file> <output file>
X_test_fpath = sys.argv[1]
dict_fpath = sys.argv[2]
output_fpath = sys.argv[3]
w2v_fpath = sys.argv[5]
model_fpath = sys.argv[6]
print('# [Info] Argv')
print('    - X test file  : {}'.format(X_test_fpath))
print('    - Dict file    : {}'.format(dict_fpath))
print('    = Output file  : {}'.format(output_fpath))
print('    - W2V file     : {}'.format(w2v_fpath))
print('    - Model file   : {}'.format(model_fpath))

def load_X(fpath):
    data = pd.read_csv(fpath)
    return np.array(data['comment'].values, dtype=str)

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
    print('# [Info] Loading W2V model...')
    w2v_model.load(w2v_fpath)
    print('# [Info] Converting texts to vectors...')
    X_train = np.zeros((len(X_seg), MAX_LENGTH, EMBEDDING_DIM))
    for n in range(len(X_seg)):
        for i in range(min(len(X_seg[n]), MAX_LENGTH)):
            try:
                vector = w2v_model[X_seg[n][i]]
                X_train[n][i] = vector
                #X_train[n][i] = (vector - vector.mean(0)) / (vector.std(0) + 1e-20)
            except KeyError as e:
                pass
                # print ('Word', X_seg[n][i], 'is not in dictionary.')
    return X_train

print('# [Info] Loading testing data...')
X_test = load_X(X_test_fpath)
print('# [Info] Loading txt dict...')
X_seg = segment(X_test)
print('# [Info] Word to vector...')
X_test = w2v(X_seg)
print('# [Info] Loading model...')
model = load_model(model_fpath)

prediction = model.predict(X_test)
print('# [Info] Output...')
with open(output_fpath, 'w') as f:
    f.write('id,label\n')
    for i, v in enumerate(prediction):
        f.write('%d,%d\n' %(i, np.argmax(v)))
print('Done!')
