import sys, os, argparse
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import UpSampling2D, Conv2D, MaxPooling2D
from keras.layers import Flatten, Reshape
from keras.layers import Dense, BatchNormalization
from keras.models import load_model

from PIL import Image
from sklearn.cluster import KMeans, Birch
from sklearn.decomposition import PCA

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

def split_train_val(X, val_ratio, shuffle=False):
    if shuffle == True:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
    train_len = int(X.shape[0] * (1.0 - val_ratio))
    return X[:train_len], X[train_len:]

def build_train_model_conv(args, data):
    # Build layers
    input_img = Input(shape=(32, 32, 3))
    x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last')(input_img)
    x = MaxPooling2D((2, 2), padding='same', data_format='channels_last')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last')(x)
    encoded = MaxPooling2D((2, 2), padding='same', data_format='channels_last')(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last')(encoded)
    x = UpSampling2D((2, 2), data_format='channels_last')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last')(x)
    x = UpSampling2D((2, 2), data_format='channels_last')(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same',data_format="channels_last")(x)

    # Compile and train
    encoder = Model(input=input_img, output=encoded)
    autoencoder = Model(input=input_img, output=decoded)
    autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(data, data, epochs=args.epoch, batch_size=args.batch, verbose=1, shuffle=True)
    # Save models
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

    # My encoder
    print('# [Info] My encoder...')
    encoder = load_model(args.encoder)
    encoded_data = encoder.predict(data)
    print(encoded_data.shape)
    encoded_data = encoded_data.reshape(len(data),-1)
    print(encoded_data.shape)

    # pca
    print('# [Info] PCA...')
    pca = PCA(n_components=200, whiten=True, random_state=0)
    encoded_data = pca.fit_transform(encoded_data)
    print(encoded_data.shape)

    # kmeans
    print('# [Info] Clustering (kmeans)...')
    kmeans = KMeans(init='k-means++', n_clusters=2, max_iter=2000, random_state=0, n_jobs=8).fit(encoded_data)
    labels = kmeans.predict(encoded_data)
    for i in range(20):
        print('{}: {}'.format(i+1, labels[i]))

    '''
    # birch
    birch = Birch(n_clusters=2, branching_factor=50)
    birch.fit(encoded_data)
    labels = birch.labels_
    '''

    # Check if labels are the same
    print('# [Info] Making prediction: {}'.format(len(test_id)))
    answers = []
    for i in range(len(test_id)):
        if labels[test_image1[i]-1] == labels[test_image2[i]-1]:
            answers.append(1)
        else:
            answers.append(0)

    # Output results
    print('# [Info] Output prediction: {}'.format(prediction_fpath))
    with open(prediction_fpath, 'w') as f:
        f.write('id,label\n')
        for i, ans in zip(test_id, answers):
            f.write('%d,%d\n' %(i, ans))

# def autoencoder(args, data):
#     autoencoder = load_model(args.autoencoder)
#     autoencoded_data = autoencoder.predict(data)
#     indices = list(range(32))
#     import matplotlib.pyplot as plt
#     # Original images
#     for i, index in enumerate(indices):
#         plt.subplot(4, 8, i+1)
#         plt.xticks([], [])
#         plt.yticks([], [])
#         plt.imshow(data[index])
#     plt.savefig('Qb1_ori.jpg')
#     plt.show()
#     plt.close()
#     # Reconstructed images
#     for i, index in enumerate(indices):
#         plt.subplot(4, 8, i+1)
#         plt.xticks([], [])
#         plt.yticks([], [])
#         plt.imshow(autoencoded_data[index])
#     plt.savefig('Qb1_rec.jpg')
#     plt.show()

# def visualize():
#     import matplotlib.pyplot as plt

#     data = np.load('../data/hw7/visualization.npy')
#     data = data.reshape(-1,32*32*3)
#     # pca
#     print('# [Info] PCA...')
#     pca = PCA(n_components=2, whiten=True, random_state=0)
#     encoded_data = pca.fit_transform(data)
#     print(encoded_data.shape)

#     # kmeans
#     print('# [Info] Clustering (kmeans)...')
#     kmeans = KMeans(init='k-means++', n_clusters=2, max_iter=2000, random_state=0, n_jobs=8).fit(encoded_data)
#     labels = kmeans.labels_

#     true_labels = [ 1 for i in range(2500) ] + [ 0 for i in range(2500) ]

#     error = 0
#     for i in range(5000):
#         if true_labels[i] != labels[i]:
#             error += 1
#     print(error)

#     label0_x, label0_y = [], []
#     label1_x, label1_y = [], []
#     for i, (x, y) in enumerate(encoded_data[:,:2]):
#         if labels[i] == 0: # change to true_labels
#             label0_x.append(x)
#             label0_y.append(y)
#         else:
#             label1_x.append(x)
#             label1_y.append(y)

#     l0 = plt.scatter(label0_x, label0_y, c='blue', marker='.')
#     l1 = plt.scatter(label1_x, label1_y, c='red', marker='.')
#     plt.legend((l0, l1), ('Dataset A', 'Dataset B'), scatterpoints=1, loc='upper right', fontsize=8)
#     plt.savefig('my_labels.jpg') # change to 'true_labels.jpg'
#     plt.show()

def main(args):
    # Fix random seeds
    np.random.seed(0)

    preprocess = Preprocess(args.image_dir, args)
    images = preprocess.get_images()
    # np.save('images.npy', images)
    # images = np.load('images.npy')
    # build_train_model_conv(args, images)
    predict(args, images)
    # autoencoder(args, images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type=str, help='[Input] Your training image directory')

    parser.add_argument('--encoder', default=None, type=str, help='[In/Output] Your encoder model')
    parser.add_argument('--autoencoder', default=None, type=str, help='[In/Output] Your autoencoder model')
    parser.add_argument('--test', default=None, type=str, help='[Input] Your testing file')
    parser.add_argument('--prediction', default=None, type=str, help='[Output] Your prediction file')

    parser.add_argument('--batch', default=512, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    args = parser.parse_args()
    print(args)
    main(args)
    print('Done!')
