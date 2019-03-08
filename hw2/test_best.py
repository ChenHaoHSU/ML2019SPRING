##############################################################
# Filename   [ test_best.py ]
# Synopsis   []
# Author     [ Chen-Hao Hsu ]
# Date       [ 2019/02/24 created ]
##############################################################

import sys
import csv
import pickle
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

def load_test(filename):
  test_X = pd.read_csv(filename, encoding='big5').values
  return test_X

def write_output(filename, test_y):
  with open(filename, 'w') as csvfile:
    fieldnames = ['id', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(test_y.shape[0]):
      writer.writerow({'id': (i+1), 'label': test_y[i]})

test_file = sys.argv[1]
output_file = sys.argv[2]
model_file = sys.argv[3]
print('[Info] Test: {}'.format(test_file))
print('[Info] Model: {}'.format(model_file))
print('[Info] Output: {}'.format(output_file))

test_X = load_test(test_file)
model = pickle.load(open(model_file, 'rb'))
test_y = model.predict(test_X)

write_output(output_file, test_y)
