import sys, os, random
import numpy as np
import pandas as pd
from keras.models import load_model
import jieba
from gensim.models import Word2Vec

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
    print('# [Info] Loading JIEBA...')
    jieba.load_userdict(dict_fpath)
    X_segment = []
    for i, sent in enumerate(X_train):
        print('\r# [Info] Segmenting sentences... {} / {}'.format(i+1, len(X_train)), end='', flush=True)
        word_list = []
        for word in list(jieba.cut(sent, cut_all=False)):
            if word[0] == 'B': continue
            word_list.append(word)
        X_segment.append(word_list)
    print('', flush=True)
    return X_segment
 
def word_to_vector(X_segment):
    print('# [Info] Building W2V model...')
    embed = Word2Vec.load(w2v_fpath)
    X_train = np.zeros((len(X_segment), MAX_LENGTH, EMBEDDING_DIM))
    for i in range(len(X_segment)):
        print('\r# [Info] Converting texts to vectors... {} / {}'.format(i+1, len(X_segment)), end='', flush=True)
        for j in range(min(len(X_segment[i]), MAX_LENGTH)):
            try:
                vector = embed[X_segment[i][j]]
                X_train[i][j] = (vector - vector.mean(0)) / (vector.std(0) + 1e-20)
            except KeyError as e:
                pass
    print('', flush=True)
    return X_train

''' Load testing data '''
X_test = load_X(X_test_fpath)
print('# [Info] {} testing data loaded.'.format(len(X_test)))

''' Preprocess '''
X_segment = text_segmentation(X_test)
X_test = word_to_vector(X_segment)

''' Prediction and Output '''
# print('# [Info] Loading model...')
# model = load_model(model_fpath)
# prediction = model.predict(X_test)
# print('# [Info] Output \'{}\'...'.format(output_fpath))
# with open(output_fpath, 'w') as f:
#     f.write('id,label\n')
#     for i, v in enumerate(prediction):
#         f.write('%d,%d\n' %(i, np.argmax(v)))
# print('Done!')

print('# [Info] Loading model...')
total = np.zeros((len(X_test, 2)))
model_names = ['kaggle74460.h5', 'kaggle74990.h5', 'kaggle73920.h5', 'kaggle74880.h5', 'kaggle75320.h5']
for i, model_name in enumerate(model_names):
    model = load_model(model_name)
    prediction = model.predict(X_test)
    for j, v in enumerate(prediction):
        total[j][np.argmax(v)] += 1
print('# [Info] Output \'{}\'...'.format(output_fpath))
with open(output_fpath, 'w') as f:
    f.write('id,label\n')
    for i, v in enumerate(total):
        f.write('%d,%d\n' %(i, np.argmax(v)))
print('Done!')
