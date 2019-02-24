##############################################################
# Filename   [ train.py ]
# Synopsis   [ Implementation of Linear Regression,
#              Output a weight file (.npy) ]
# Author     [ Chen-Hao Hsu ]
# Date       [ 2019/02/22 created ]
##############################################################

import sys
import csv
import numpy as np
import pandas as pd

def load_train(filename):
  raw_data = pd.read_csv(filename, encoding='big5').as_matrix()
  data = raw_data[:, 3:] # 12 months, 20 days per month, 18 features per day. shape: (4320 , 24)
  data[data == 'NR'] = 0.0
  data = data.astype('float')
  X, Y = [], []
  for i in range(0, data.shape[0], 18*20):
      # i: start of each month
      days = np.vsplit(data[i:i+18*20], 20) # shape: 20 * (18, 24)
      concat = np.concatenate(days, axis=1) # shape: (18, 480)
      for j in range(0, concat.shape[1]-9):
        X.append(concat[:, j:j+9].flatten())
        Y.append([concat[9, j+9]])
  train_X = np.array(X)
  train_y = np.array(Y)
  train_X = np.c_[ train_X, np.ones(train_X.shape[0]) ] # add the bias
  return train_X, train_y

def ada_grad(train_X, train_y):
  # initialize parameters
  w = np.ones((train_X.shape[1], 1)) # initial weight
  lr = 1.0
  sum_grad = 0.0
  iteration = 100000

  # iterations
  for i in range(iteration):
    print('iteration {}'.format(i))
    pred_y = np.dot(train_X, w)
    loss = pred_y - train_y
    cur_grad = 2 * np.dot(np.transpose(train_X), loss)
    sum_grad += cur_grad ** 2
    ada = np.sqrt(sum_grad)
    w -= lr / ada * cur_grad
  return w

########
# Main #
########
train_file = sys.argv[1]
weight_file = sys.argv[2]
train_X, train_y = load_train(train_file)
w = ada_grad(train_X, train_y)
np.save(weight_file, w)
