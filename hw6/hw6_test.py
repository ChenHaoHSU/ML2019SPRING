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
X_test_segment = text_segmentation(X_test)
embed = Word2Vec.load(w2v_fpath)
X_test = word_to_vector(X_test_segment)

''' Prediction and Output '''
print('# [Info] Load model: {}'.format(model_fpath))
model = load_model(model_fpath)
prediction = model.predict(X_test)
print('# [Info] Output prediction: {}'.format(output_fpath))
with open(output_fpath, 'w') as f:
    f.write('id,label\n')
    for i, v in enumerate(prediction):
        f.write('%d,%d\n' %(i, np.argmax(v)))
print('Done!')
