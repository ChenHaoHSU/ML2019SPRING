import sys, os, random
import gc
import numpy as np
import pandas as pd
from keras.models import load_model
from keras import backend as K
import jieba
from gensim.models import Word2Vec
import emoji

# bash hw6_test.sh <test_x file> <dict.txt.big file> <output file>
X_test_fpath = sys.argv[1]
dict_fpath = sys.argv[2]
output_fpath = sys.argv[3]
w2v_fpath = sys.argv[4]
model_fpaths = ['model_0.h5', 'model_1.h5', 'model_2.h5', 'model_3.h5', 'model_4.h5', 'model_5.h5',\
                'model_6.h5', 'model_7.h5', 'model_8.h5', 'model_9.h5', 'model_10.h5']
print('# [Info] Argv')
print('    - X test file  : {}'.format(X_test_fpath))
print('    - Dict file    : {}'.format(dict_fpath))
print('    = Output file  : {}'.format(output_fpath))
print('    - W2V file     : {}'.format(w2v_fpath))
print('    - Model file   : {}'.format(model_fpaths))

''' Fix random seeds '''
random.seed(0)
np.random.seed(0)

MAX_LENGTH = 40
EMBEDDING_DIM = 200

BATCH_SIZE = 500

def load_X(fpath):
    data = pd.read_csv(fpath)
    return np.array(data['comment'].values, dtype=str)

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

''' Load testing data '''
X_test = load_X(X_test_fpath)
print('# [Info] {} testing data loaded.'.format(len(X_test)))

''' Preprocess '''
print('# [Info] Loading JIEBA...')
jieba.load_userdict(dict_fpath)
X_test_segment = text_segmentation(X_test)
embed = Word2Vec.load(w2v_fpath)
X_test = word_to_vector(embed, X_test_segment)

''' Prediction and Output '''
total_votes = np.zeros((len(X_test), 2))
for model_fpath in model_fpaths:
    print('# [Info] Load model: {}'.format(model_fpath))
    model = load_model(model_fpath)
    pred = model.predict(X_test, batch_size=BATCH_SIZE)
    for i, v in enumerate(pred):
        total_votes[i][np.argmax(v)] += 1
    del model
    model = None
    gc.collect
    K.clear_session()
    print(total_votes)

''' Output prediction '''
print('# [Info] Output prediction: {}'.format(output_fpath))
with open(output_fpath, 'w') as f:
    f.write('id,label\n')
    for i, v in enumerate(total_votes):
        f.write('%d,%d\n' %(i, np.argmax(v)))
print('Done!')
