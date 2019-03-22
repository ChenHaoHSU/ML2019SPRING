### [1]
import numpy as np

import sys
X_train_fpath = sys.argv[1]
Y_train_fpath = sys.argv[2]
X_test_fpath = sys.argv[3]
output_fpath = sys.argv[4]

### [3]
selected_columns = None
X_train = np.genfromtxt(X_train_fpath, delimiter=',', skip_header=1, usecols=selected_columns)
Y_train = np.genfromtxt(Y_train_fpath, delimiter=',', skip_header=1)

### [4]
def _normalize_column_0_1(X, train=True, specified_column = None, X_min = None, X_max=None):
    # The output of the function will make the specified column of the training data 
    # from 0 to 1
    # When processing testing data, we need to normalize by the value 
    # we used for processing training, so we must save the max value of the 
    # training data
    if train:
        if specified_column == None:
            specified_column = np.arange(X.shape[1])
        length = len(specified_column)
        X_max = np.reshape(np.max(X[:, specified_column], 0), (1, length))
        X_min = np.reshape(np.min(X[:, specified_column], 0), (1, length))
        
    X[:, specified_column] = np.divide(np.subtract(X[:, specified_column], X_min), np.subtract(X_max, X_min))

    return X, X_max, X_min

### [5]
def _normalize_column_normal(X, train=True, specified_column = None, X_mean=None, X_std=None):
    # The output of the function will make the specified column number to 
    # become a Normal distribution
    # When processing testing data, we need to normalize by the value 
    # we used for processing training, so we must save the mean value and 
    # the variance of the training data
    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        length = len(specified_column)
        X_mean = np.reshape(np.mean(X[:, specified_column],0), (1, length))
        X_std  = np.reshape(np.std(X[:, specified_column], 0), (1, length))
    
    X[:,specified_column] = np.divide(np.subtract(X[:,specified_column],X_mean), X_std)
     
    return X, X_mean, X_std

### [9]
def accuracy(Y_pred, Y_label):
    assert Y_pred.shape == Y_label.shape
    acc = np.sum(Y_pred == Y_label)/len(Y_pred)
    return acc

col = None
X_train, X_mean, X_std = _normalize_column_normal(X_train, specified_column=col)
X_test = np.genfromtxt(X_test_fpath, delimiter=',', skip_header=1, usecols=selected_columns)
X_test, _, _= _normalize_column_normal(X_test, train=False, specified_column=col, X_mean=X_mean, X_std=X_std)

###############################
# RandomForest
###############################
from sklearn.ensemble import RandomForestClassifier
import pickle
import random

random.seed(0)

model = RandomForestClassifier(n_estimators=500, max_depth=20, n_jobs=-1)
model.fit(X_train, Y_train)

train_prediction = model.predict(X_train)
train_acc = accuracy(train_prediction, Y_train)
print('Train Acc:', train_acc)

# model_fpath = 'model/model_randomforest.pickle'
# pickle.dump(model, open(model_fpath, 'wb'))

test_prediction = model.predict(X_test)
with open(output_fpath, 'w') as f:
    f.write('id,label\n')
    for i, v in  enumerate(test_prediction):
        f.write('%d,%d\n' %(i+1, int(v)))

###############################
# Keras
###############################
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation
# from keras.layers import Conv2D, MaxPooling2D, Flatten
# from keras.optimizers import SGD, Adam
# from keras.utils import np_utils, to_categorical

# Y_train = to_categorical(Y_train)

# model = Sequential()
# model.add(Dense(input_dim=X_train.shape[1], units=500, activation='relu'))
# for i in range(10):
#     model.add(Dense(units=500, activation='relu'))
# model.add(Dense(units=2, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
# model.fit(X_train, Y_train, batch_size=100, epochs=50)
# result = model.evaluate(X_train, Y_train)
# print('\nTrain Acc:', result[1])

# prediction = model.predict(X_test)

# with open(output_fpath, 'w') as f:
#     f.write('id,label\n')
#     for i, v in enumerate(prediction):
#         f.write('%d,%d\n' %(i+1, np.argmax(v)))
