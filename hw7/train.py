import sys, os, argparse
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Dropout
from keras.layers import BatchNormalization
from keras.utils import np_utils, to_categorical

from PIL import Image

import matplotlib.pyplot as plt
from scipy.misc import imsave

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

def build_model(args, data):
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
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(data, data, epochs=args.epoch, batch_size=args.batch, verbose=1)
    autoencoder.summary()
    # Save Model
    autoencoder.save(args.autoencoder_fpath)
    encoder.save(args.encoder_fpath)

def main(args):
    #preprocess = Preprocess(args.image_dir, args)
    #flatten_images = preprocess.get_flatten_images()
    #np.save('flatten_images.npy', flatten_images)

    flatten_images = np.load('flatten_images.npy')
    build_model(args, flatten_images)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type=str, help='[Input] Your training image directory')
    parser.add_argument('encoder_fpath', type=str, help='[Output] Your encoder model')
    parser.add_argument('autoencoder_fpath', type=str, help='[Output] Your autoencoder model')

    parser.add_argument('--latent_dim', default=32, type=int)
    parser.add_argument('--batch', default=256, type=int)
    parser.add_argument('--epoch', default=50, type=int)
    args = parser.parse_args()
    print(args)
    main(args)
