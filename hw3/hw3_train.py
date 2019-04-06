# ## Import package
# import sys
# import csv
# import time
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader

# ## Read in train.csv and split data into training/validation set

# train_fpath = sys.argv[1]
# model_fpath = sys.argv[2]

# def readfile(path):
#     print("Reading File...")
#     x_train = []
#     x_label = []
#     val_data = []
#     val_label = []

#     raw_train = np.genfromtxt(path, delimiter=',', dtype=str, skip_header=1)
#     for i in range(len(raw_train)):
#         tmp = np.array(raw_train[i, 1].split(' ')).reshape(1, 48, 48)
#         if (i % 10 == 0):
#             val_data.append(tmp)
#             val_label.append(raw_train[i][0])
#         else:
#             x_train.append(tmp)
#             x_train.append(np.flip(tmp, axis=2))    # simple example of data augmentation
#             x_label.append(raw_train[i][0])
#             x_label.append(raw_train[i][0])

#     x_train = np.array(x_train, dtype=float) / 255.0
#     val_data = np.array(val_data, dtype=float) / 255.0
#     x_label = np.array(x_label, dtype=int)
#     val_label = np.array(val_label, dtype=int)
#     x_train = torch.FloatTensor(x_train)
#     val_data = torch.FloatTensor(val_data)
#     x_label = torch.LongTensor(x_label)
#     val_label = torch.LongTensor(val_label)

#     return x_train, x_label, val_data, val_label

# x_train, x_label, val_data, val_label = readfile(train_fpath)    # 'train.csv'

# ## Wrapped as dataloader
# train_set = TensorDataset(x_train, x_label)
# val_set = TensorDataset(val_data, val_label)

# batch_size = 256
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
# val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)

# ## Model Construction
# def gaussian_weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1 and classname.find('Conv') == 0:
#         m.weight.data.normal_(0.0, 0.02)


# class Classifier(nn.Module):
#     def __init__(self):
#         super(Classifier, self).__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, 64, 4, 2, 1),  # [64, 24, 24]
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 64, 3, 1, 1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2),
#             nn.MaxPool2d(2, 2, 0),      # [64, 12, 12]

#             nn.Conv2d(64, 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(128, 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),
#             nn.MaxPool2d(2, 2, 0),      # [128, 6, 6]

#             nn.Conv2d(128, 256, 3, 1, 1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(256, 256, 3, 1, 1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#             nn.MaxPool2d(2, 2, 0)       # [256, 3, 3]
#         )

#         self.fc = nn.Sequential(
#             nn.Linear(256*3*3, 1024),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(p=0.5),
#             nn.Linear(1024, 512),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(p=0.5),
#             nn.Linear(512, 7)
#         )

#         self.cnn.apply(gaussian_weights_init)
#         self.fc.apply(gaussian_weights_init)

#     def forward(self, x):
#         out = self.cnn(x)
#         out = out.view(out.size()[0], -1)
#         return self.fc(out)

# ## Training
# model = Classifier().cuda()
# # print(model)
# loss = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# best_acc = 0.0

# for epoch in range(num_epoch):
#     epoch_start_time = time.time()
#     train_acc = 0.0
#     train_loss = 0.0
#     val_acc = 0.0
#     val_loss = 0.0

#     model.train()
#     for i, data in enumerate(train_loader):
#         optimizer.zero_grad()

#         train_pred = model(data[0].cuda())
#         batch_loss = loss(train_pred, data[1].cuda())
#         batch_loss.backward()
#         optimizer.step()

#         train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
#         train_loss += batch_loss.item()

#         progress = ('#' * int(float(i)/len(train_loader)*40)).ljust(40)
#         print ('[%03d/%03d] %2.2f sec(s) | %s |' % (epoch+1, num_epoch, \
#                 (time.time() - epoch_start_time), progress), end='\r', flush=True)
    
#     model.eval()
#     for i, data in enumerate(val_loader):
#         val_pred = model(data[0].cuda())
#         batch_loss = loss(val_pred, data[1].cuda())

#         val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
#         val_loss += batch_loss.item()

#         progress = ('#' * int(float(i)/len(val_loader)*40)).ljust(40)
#         print ('[%03d/%03d] %2.2f sec(s) | %s |' % (epoch+1, num_epoch, \
#                 (time.time() - epoch_start_time), progress), end='\r', flush=True)

#     val_acc = val_acc/val_set.__len__()
#     print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
#             (epoch + 1, num_epoch, time.time()-epoch_start_time, \
#              train_acc/train_set.__len__(), train_loss, val_acc, val_loss))

#     if (val_acc > best_acc):
#         with open('acc.txt','w') as f:
#             f.write(str(epoch)+'\t'+str(val_acc)+'\n')
#         torch.save(model.state_dict(), model_fpath)
#         best_acc = val_acc
#         print ('Model Saved!')

####################################################################
####################################################################
####################################################################

import sys
import csv
import math
import pickle
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dense, Dropout, Activation
from keras.layers import ZeroPadding2D, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils, to_categorical
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator

def load_train(train_fpath):
    data = pd.read_csv(train_fpath)
    Y_train = np.array(data['label'].values, dtype=int)
    X_train = []
    for features in data['feature'].values:
        split_features = [ int(i) for i in features.split(' ') ]
        matrix_features = np.array(split_features).reshape(48, 48, 1)
        X_train.append(matrix_features)
    X_train = np.array(X_train)
    return X_train, Y_train

train_fpath = sys.argv[1]
model_fpath = sys.argv[2]
print('# Train : {}'.format(train_fpath))
print('# Model : {}'.format(model_fpath))

X_train, Y_train = load_train(train_fpath)
Y_train = np_utils.to_categorical(Y_train, 7)
print(Y_train.shape)

# ImageDataGenerator
datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.5,
	rotation_range=13,
    horizontal_flip=False,
    fill_mode='nearest')
datagen.fit(X_train)

model = Sequential()
# CNN
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)))
model.add(ZeroPadding2D(padding=(2, 2), data_format='channels_last'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

for i in range(2):
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

model.add(Flatten())

# DNN
for i in range(1):
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
for i in range(1):
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
for i in range(1):
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
for i in range(1):
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
model.add(Dense(units=7, activation='softmax'))

model.summary()

batch_size = 200
epochs = 100
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
# model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)
train_history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size,shuffle=True),
                    steps_per_epoch=3 * (math.floor(len(X_train)/batch_size)), epochs=epochs,
                    verbose=1)

result = model.evaluate(X_train, Y_train)
print('\nTrain Acc:', result[1])

model.save(model_fpath)
