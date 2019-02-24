##############################################################
# Filename   [ train.py ]
# Synopsis   [ Implementation of Linear Regression ]
# Author     [ Chen-Hao Hsu ]
# Date       [ 2019/02/22 created ]
##############################################################

import sys
import csv
import numpy as np
import pandas as pd
from argparse import ArgumentParser

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
  return np.array(X), np.array(Y)

def load_test(filename):
    raw_data = pd.read_csv(filename, header=None, encoding='big5').as_matrix()
    data = raw_data[:, 2:]
    data[data == 'NR'] = 0.0
    data = data.astype('float')
    obs = np.vsplit(data, data.shape[0]/18)
    X = []
    for i in obs:
        X.append(i.flatten())
    return np.array(X)

def ada_grad(train_X, train_y):
  N = train_X.shape[0]
  features_num = train_X.shape[1]

  # initialize parameters
  w = np.ones((train_X.shape[1], 1)) # initial w
  lr = 1.0
  total_grad = 0.0
  iteration = 200000

  # Iterations
  for i in range(iteration):
    print('iter = {}'.format(i))
    pred_y = np.dot(train_X, w)
    loss = pred_y - train_y
    w_grad = 2 * np.dot(np.transpose(train_X), loss)
    total_grad += w_grad ** 2
    ada = np.sqrt(total_grad)
    w -= lr / ada * w_grad
  
  return w

train_file = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]

train_X, train_y = load_train(train_file)

train_X = np.c_[ train_X, np.ones(train_X.shape[0]) ]

# x_data = [ 338., 333., 328., 207., 226., 25., 179., 60., 208.,  606.] 
# y_data = [[ 640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]]
# train_X = np.array(x_data)
# train_X = np.transpose(train_X)
# train_X = np.c_[ train_X, np.ones(train_X.shape[0]) ]
# train_y = np.array(y_data)
# train_y = np.transpose(train_y)

w = ada_grad(train_X, train_y)

test_X = load_test(test_file)
test_X = np.c_[ test_X, np.ones(test_X.shape[0]) ]

test_y = np.dot(test_X, w)
print(test_y)

with open(output_file, 'w') as csvfile:
    fieldnames = ['id', 'value']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(test_y.shape[0]):
      writer.writerow({'id': 'id_{}'.format(i), 'value': str(test_y[i][0])})

