##############################################################
# Filename   [ train_best.py ]
# Synopsis   [ Implementation of Linear Regression,
#              Output a weight file (.npy) ]
# Author     [ Chen-Hao Hsu ]
# Date       [ 2019/03/08 created ]
##############################################################

import sys
import pickle
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression

def load_train(train_X_file, train_y_file):
  train_X = pd.read_csv(train_X_file, encoding='big5').values
  train_y = pd.read_csv(train_y_file, encoding='big5').values.flatten()
  return train_X, train_y

########
# Main #
########
train_X_file = sys.argv[1]
train_y_file = sys.argv[2]
model_file = sys.argv[3]
train_X, train_y = load_train(train_X_file, train_y_file)

# model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2))
# model = svm.SVC(gamma='scale')
# model = linear_model.SGDClassifier(loss='log', learning_rate='adaptive', max_iter=1000, tol=1e-3)
model = LogisticRegression(random_state=0, solver='lbfgs', max_iter=100000)
model.fit(train_X, train_y)
pickle.dump(model, open(model_file, 'wb'))
