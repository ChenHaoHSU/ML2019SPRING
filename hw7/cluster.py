import sys, os, argparse
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Dropout
from keras.layers import BatchNormalization
from keras.utils import np_utils, to_categorical
from keras.models import load_model

from PIL import Image
from sklearn import *

class Preprocess():
    def __init__(self, image_dir, args):
        self.images = self.load_images(image_dir)
        self.images = self.images / 255.0 # data normalization

    def load_images(self, image_dir):
        image_num = 40000
        images = np.zeros((image_num, 32, 32, 3))
        for i in range(image_num):
            image_file = os.path.join(image_dir, '{:06d}.jpg'.format(i+1))
            print('\r# [Info] Load images: {}'.format(image_file), end='', flush=True)
            im = Image.open(image_file)
            images[i] = np.array(im)
        print('', flush=True)
        return images

    def get_images(self):
        return self.images

    def get_flatten_images(self):
        shape = self.images.shape
        flatten_images = np.zeros((shape[0], shape[1]*shape[2]*shape[3]))
        for i in range(shape[0]):
            flatten_images[i] = self.images[i].flatten()
        return flatten_images

def build_train_model(args, data):
    # Encoder
    input_img = Input(shape=(32*32*3,))
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    encoded_output = Dense(args.latent_dim)(encoded)
    # Decoder
    decoded = Dense(32, activation='relu')(encoded_output)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(32*32*3, activation='tanh')(decoded)
    # Build Encoder
    encoder = Model(input=input_img, output=encoded_output)
    # Build Autoencoder
    autoencoder = Model(input=input_img, output=decoded)
    autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(data, data, epochs=args.epoch, batch_size=args.batch, verbose=1, shuffle=True)
    # Save Model
    encoder.save(args.encoder)
    print('# [Info] Encoder model saved: {}'.format(args.encoder))
    autoencoder.save(args.autoencoder)
    print('# [Info] Autoencoder model saved: {}'.format(args.autoencoder))

def load_test(path):
    test = pd.read_csv(path)
    test_id = np.array(test.id)
    test_image1 = np.array(test.image1_name, dtype=int)
    test_image2 = np.array(test.image2_name, dtype=int)
    return test_id, test_image1, test_image2

def predict(args, data):
    test_fpath = args.test
    prediction_fpath = args.prediction

    print('# [Info] Loading test: {}'.format(test_fpath))
    test_id, test_image1, test_image2 = load_test(test_fpath)

    print('# [Info] Clustering...')
    encoder = load_model(args.encoder)
    encoded_data = encoder.predict(data)
    clf = cluster.KMeans(init='k-means++', n_clusters=2, random_state=0)
    clf.fit(encoded_data)
    predict = clf.predict(encoded_data)

    print('# [Info] Making prediction: {}'.format(len(test_id)))
    labels = []
    for i in range(len(test_id)):
        if predict[test_image1[i]-1] == predict[test_image2[i]-1]:
            labels.append(1)
        else:
            labels.append(0)

    print('# [Info] Output prediction: {}'.format(prediction_fpath))
    with open(prediction_fpath, 'w') as f:
        f.write('id,label\n')
        for id, label in zip(test_id, labels):
            f.write('%d,%d\n' %(id, label))

def main(args):
    # preprocess = Preprocess(args.image_dir, args)
    # flatten_images = preprocess.get_flatten_images()
    # np.save('flatten_images.npy', flatten_images)
    flatten_images = np.load('flatten_images.npy')

    TEST_ONLY = True
    # TEST_ONLY = False
    if TEST_ONLY == False:
        build_train_model(args, flatten_images)
    predict(args, flatten_images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type=str, help='[Input] Your training image directory')

    parser.add_argument('--encoder', default=None, type=str, help='[In/Output] Your encoder model')
    parser.add_argument('--autoencoder', default=None, type=str, help='[In/Output] Your autoencoder model')
    parser.add_argument('--test', default=None, type=str, help='[Input] Your testing file')
    parser.add_argument('--prediction', default=None, type=str, help='[Output] Your prediction file')

    parser.add_argument('--latent_dim', default=128, type=int)
    parser.add_argument('--batch', default=128, type=int)
    parser.add_argument('--epoch', default=50, type=int)
    args = parser.parse_args()
    print(args)
    main(args)
    print('Done!')
