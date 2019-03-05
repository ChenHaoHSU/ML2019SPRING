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
  csv_data = pd.read_csv(filename, header=None, encoding='big5').values
  test_id = np.array([ csv_data[i][0] for i in range(0, csv_data.shape[0], 18)])
  data = csv_data[:, 2:]
  data[data == 'NR'] = 0.0
  data = data.astype('float')
  id_num = test_id.shape[0]
  id_data = np.vsplit(data, id_num)
  X = []
  duration = 9
  for one_id_data in id_data:
    sqrPM25 = [[ a**2 for a in one_id_data[9]]]
    one_id_data = np.concatenate((one_id_data, sqrPM25))
    X.append(one_id_data[:, -duration:].flatten())
    # X.append(np.array([ one_id_data[i, -duration:] for i in [2,5,7,8,9,12] ]).flatten())
  test_X = np.array(X)
  test_X = np.c_[ test_X, np.ones(test_X.shape[0]) ] # add the bias
  return test_id, test_X

def write_output(filename, test_id, test_y):
  assert test_id.shape[0] == test_y.shape[0]
  with open(filename, 'w') as csvfile:
    fieldnames = ['id', 'value']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(test_y.shape[0]):
      writer.writerow({'id': test_id[i], 'value': str(test_y[i][0])})

test_file = sys.argv[1]
output_file = sys.argv[2]
weight_file = sys.argv[3]
print('[Info] Test: {}'.format(test_file))
print('[Info] Weight: {}'.format(weight_file))
print('[Info] Output: {}'.format(output_file))

test_id, test_X = load_test(test_file)
w = np.load(weight_file)
test_y = np.dot(test_X, w)

write_output(output_file, test_id, test_y)
