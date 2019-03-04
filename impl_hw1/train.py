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
  data = pd.read_csv(filename, encoding='big5').values[:, 3:]
  data[data == 'NR'] = 0.0
  data = data.astype('float')
  X, Y = [], []
  duration = 5
  month_data = np.vsplit(data, 12) # 12 months; 20 days per month
  for one_month_data in month_data:
    hour_data = np.vsplit(one_month_data, 20)
    concat_hour_data = np.concatenate(hour_data, axis=1)
    for i in range(len(concat_hour_data[0])-duration):
      X.append(concat_hour_data[:, i:i+duration].flatten()) # previous 9 (duration) hours data
      Y.append([concat_hour_data[9, i+duration]])
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
