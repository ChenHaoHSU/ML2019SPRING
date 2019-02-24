##############################################################
# Filename   [ test.py ]
# Synopsis   []
# Author     [ Chen-Hao Hsu ]
# Date       [ 2019/02/24 created ]
##############################################################

import sys
import csv
import numpy as np
import pandas as pd

def load_test(filename):
  raw_data = pd.read_csv(filename, header=None, encoding='big5').as_matrix()
  data = raw_data[:, 2:]
  data[data == 'NR'] = 0.0
  data = data.astype('float')
  obs = np.vsplit(data, data.shape[0]/18)
  X = []
  for i in obs:
     X.append(i.flatten())
  test_X = np.array(X)
  test_X = np.c_[ test_X, np.ones(test_X.shape[0]) ] # add the bias
  return test_X

def write_output(filename, test_y):
  with open(filename, 'w') as csvfile:
    fieldnames = ['id', 'value']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(test_y.shape[0]):
      writer.writerow({'id': 'id_{}'.format(i), 'value': str(test_y[i][0])})
    print('[Info] Output: {}'.format(filename))

test_file = sys.argv[1]
output_file = sys.argv[2]
weight_file = sys.argv[3]

test_X = load_test(test_file)
w = np.load(weight_file)
test_y = np.dot(test_X, w)

write_output(output_file, test_y)
