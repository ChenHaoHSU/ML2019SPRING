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
w2v_fpath = sys.argv[4]
model_fpath = sys.argv[5]
print('# [Info] Argv')
print('    - X test file  : {}'.format(X_test_fpath))
print('    - Dict file    : {}'.format(dict_fpath))
print('    = Output file  : {}'.format(output_fpath))
print('    - W2V file     : {}'.format(w2v_fpath))
print('    - Model file   : {}'.format(model_fpath))

MAX_LENGTH = 40
EMBEDDING_DIM = 100

def load_X(fpath):
    data = pd.read_csv(fpath)
    return np.array(data['comment'].values, dtype=str)

def text_segmentation(X_train):
    print('# [Info] Segmenting text...')
    jieba.load_userdict(dict_fpath)
    X_segment = []
    filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n '+'～＠＃＄％︿＆＊（）！？⋯  ，。'
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
    w2v_model = word2vec.Word2Vec.load(w2v_fpath)
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

''' Load testing data '''
X_test = load_X(X_test_fpath)
print('# [Info] {} testing data loaded.'.format(len(X_test)))

''' Preprocess '''
X_segment = text_segmentation(X_test)
X_test = word_to_vector(X_segment)

''' Prediction and Output '''
print('# [Info] Loading model...')
model = load_model(model_fpath)
prediction = model.predict(X_test)
print('# [Info] Output \'{}\'...'.format(output_fpath))
with open(output_fpath, 'w') as f:
    f.write('id,label\n')
    for i, v in enumerate(prediction):
        f.write('%d,%d\n' %(i, np.argmax(v)))
print('Done!')
