import numpy as np
import csv
import sys
import argparse

def parse_train(train):
    X_train = []
    Y_train = []
    with open(train, newline='') as csvfile:
        rows = csv.reader(csvfile)
        data = list(rows)[1:]
    data = np.array(data)
    y = data[:,0]
    x = data[:,1]
    for i in range(len(x)):
        l = x[i].split(' ')
        tmp = []
        for j in range(len(l)):
            tmp.append(float(l[j]))
        X_train.append(tmp)
        Y_train.append(float(y[i]))
    X_train = np.array(X_train, dtype = 'float32')
    Y_train = np.array(Y_train, dtype = 'float32')
    print(X_train.shape, Y_train.shape)
    np.save("X_train.npy", X_train)
    np.save("Y_train.npy", Y_train)

def parse_test(test):
    X_test = []
    with open(test, newline='') as csvfile:
        rows = csv.reader(csvfile)
        test_data = list(rows)[1:]
    test_data = np.array(test_data)[:,1]
    for i in range(len(test_data)):
        l = test_data[i].split(' ')
        tmp = []
        for j in range(len(l)):
            tmp.append(float(l[j]))
        X_test.append(tmp)
    X_test = np.array(X_test, dtype = np.float)
    print(X_test.shape)
    np.save("X_test.npy", X_test)

def main(args):
    parse_train(args.train)
    parse_test(args.test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', default='../data/hw8/train.csv', dest='train', type=str, help='[Input] Your train.csv')
    parser.add_argument('-test', default='../data/hw8/test.csv', dest='test', type=str, help='[Input] Your test.csv')
    args = parser.parse_args()
    main(args)

    
