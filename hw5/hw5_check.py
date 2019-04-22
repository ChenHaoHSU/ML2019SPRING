import sys
import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.misc import imsave

input_dir = sys.argv[1]
label_fpath = 'labels.csv'
output_dir = sys.argv[2]
print('# Input dir  : {}'.format(input_dir))
print('# Label path : {}'.format(label_fpath))
print('# Output dir : {}'.format(output_dir))

def load_input(input_dir):
    X_train = []
    for i in range(200):
        image_file = os.path.join(input_dir, '{:03d}.png'.format(i))
        im = Image.open(image_file)
        im_arr = np.array(im.getdata()).reshape(im.size[0], im.size[1], 3)
        X_train.append(im_arr)
    return np.array(X_train)

def load_labels(label_fpath):
    data = pd.read_csv(label_fpath)
    Y_train = np.array(data['TrueLabel'].values, dtype=int)
    return Y_train

X_origin = load_input(input_dir)
print('# [Info] Load {} original images'.format(len(X_origin)))
X_trans = load_input(output_dir)
print('# [Info] Load {} trans images'.format(len(X_trans)))
Y_train = load_labels(label_fpath)
print('# [Info] Load {} labels'.format(len(Y_train)))

total_max = 0
for id, (origin, trans) in enumerate(zip(X_origin, X_trans)):
    diff_sum, diff_max = 0, 0
    assert origin.shape == trans.shape
    diff = trans - origin
    diff = np.absolute(diff)
    diff_max = np.max(diff)
    diff_avg = np.sum(diff) / (origin.shape[0]*origin.shape[1]*origin.shape[2])
    # print(diff_max, diff_avg)
    total_max += diff_max

print('L-inf:', total_max/200.0)
