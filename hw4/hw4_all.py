import sys
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import np_utils
import keras.backend as K

# Handle argv
train_fpath = sys.argv[1]
output_fpath = sys.argv[2]
model_fpath = sys.argv[3]
output_fpath = output_fpath if output_fpath[-1] != '/' else output_fpath[:-1]
print('# Training data : {}'.format(train_fpath))
print('# Output path   : {}'.format(output_fpath))
print('# Model         : {}'.format(model_fpath))

# Load data
def load_train(train_fpath):
    normalization = False
    data = pd.read_csv(train_fpath)
    Y_train = np.array(data['label'].values, dtype=int)
    X_train = []
    for features in data['feature'].values:
        split_features = [ int(i) for i in features.split(' ') ]
        matrix_features = np.array(split_features).reshape(48, 48, 1)
        #matrix_features = np.array(split_features).reshape(48*48)
        X_train.append(matrix_features)
    if normalization == True:
        X_train = np.array(X_train, dtype=float) / 255.0
    else:
        X_train = np.array(X_train, dtype=float)
    return X_train, Y_train

print('# Loading data...')
X_train, Y_train = load_train(train_fpath)
Y_train = np_utils.to_categorical(Y_train, 7)

# Load model
print('# Loading model...')
model = load_model(model_fpath)

###############################################
# Figure 2 (Filter Visualization)
###############################################
print('### Start Saliency Map...')
label = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral']
input_img = model.input
image_ids = [15, 299, 9, 25, 70, 81, 94]

for i, id in enumerate(image_ids):

    print('# Plotting {} ({})...'.format(label[i], id))

    # Get function
    img = X_train[id].reshape(1, 48, 48, 1)
    pred = model.predict(img).argmax(axis=-1)
    target = K.mean(model.output[:, pred[0]])
    grads = K.gradients(target, input_img)[0]
    fn = K.function([input_img, K.learning_phase()], [grads])

    # Calculate saliency map
    sliency_map = fn([img, 0])[0].reshape(48, 48)
    std, mean = np.std(sliency_map), np.mean(sliency_map)
    std, mean = sliency_map.std(), sliency_map.mean()
    heaptmap = (sliency_map - mean) / (std+1e-5)
    sliency_map -= sliency_map.mean()
    sliency_map /= sliency_map.std()

    # Plot original figure
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(12, 3), ncols=3)
    original = img.reshape(48, 48)
    img1 = ax1.imshow(original, cmap='gray')
    ax1.set_title('{}. {} ({})'.format(i, label[i], id))

    thres = sliency_map.std()
    original[np.where(abs(sliency_map) <= thres)] = original.mean()

    # Plot sliency
    img2 = ax2.imshow(sliency_map, cmap='jet')
    fig.colorbar(img2, ax=ax2)
    ax2.set_title('Saliency Map'.format(label[i], id))
    plt.tight_layout()

    # Plot mask
    img3 = ax3.imshow(original, cmap='gray')
    fig.colorbar(img3, ax=ax3)
    ax3.set_title('Mask'.format(label[i], id))
    plt.tight_layout()
    plt.savefig('{}/fig1_{}.jpg'.format(output_fpath, i))
    print('*** Save image {}/fig1_{}.jpg!'.format(output_fpath, i))
    plt.close()

###############################################
# Figure 2 (Filter Visualization)
###############################################
print('### Start Filter Visulization...')
np.random.seed(0)

# model = load_model(model_fpath)
input_img = model.input
layer_dict = dict([(layer.name, layer) for layer in model.layers])

layer_name = "conv2d_1"
print('# Visualize {}'.format(layer_name))
# filter_index = [1, 2, 3, 12, 16, 17, 23, 29, 30, 32, 36, 37, 38, 40, 46, 48,\
#                 54, 60, 62, 64, 65, 74, 78, 80, 83, 86, 92, 95, 98,\
#                 108, 112, 113, 119, 123, 127, 132, 135, 142, 145,\
#                 156, 165, 167, 171, 175, 179, 184, 187, 195, 197]
filter_index = [1, 2, 12, 16, 17, 29, 30, 32, 36, 37, 38, 40, 46,\
                54, 62, 65, 74, 78, 80, 83, 92, 95, 98, 108, 112, 113, 119, 123, 127, 132, 142, 145]
# print(len(filter_index))

########################
# fig2_1
########################
random_img = np.ones((1, 48, 48, 1))
for i, f in enumerate(filter_index):
    print('# Filter {}/{}'.format(i, len(filter_index)))
    layer_output = layer_dict[layer_name].output

    loss = K.mean(layer_output[:, :, :, f])
    grads = K.gradients(loss, input_img)[0]
    grads /= (K.sqrt(K.mean(K.square(grads)))+1e-5)
    iterate = K.function([input_img], [loss, grads])

    input_img_data = np.array(random_img)

    # Gradient ascent (AdaGrad)
    lr = 5.0
    for iter in range(500):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += lr * grads_value
        # print("\riteration: " + repr(iter) + ", current loss: " + repr(loss_value), end="", flush=True)
    # print("", flush=True)

    # Plot the visualization result
    img = input_img_data[0].reshape(48, 48)
    plt.subplot(4, 8, i+1)
    plt.title('{}'.format(i))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.imshow(input_img_data[0].reshape(48, 48), cmap='gray')
    
plt.savefig('{}/fig2_1.jpg'.format(output_fpath))
print('*** Save image {}/fig2_1.jpg!'.format(output_fpath))
plt.close()
# plt.show()

########################
# fig2_2
########################
idx = 82 # Surprised
input_img = model.input
photo = X_train[idx].reshape(1, 48, 48, 1)
fn = K.function([input_img,K.learning_phase()],[layer_dict[layer_name].output])
im = fn([photo,0])
nb_filter = im[0].shape[3]
for i, f in enumerate(filter_index):
    plt.subplot(4, 8, i+1)
    plt.title('{}'.format(i))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.imshow(im[0][0,:,:,f].reshape(48, 48), cmap='gray')

plt.savefig('{}/fig2_2.jpg'.format(output_fpath))
print('*** Save image {}/fig2_2.jpg!'.format(output_fpath))
plt.close()
# plt.show()

###############################################
# Figure 3 (Lime)
###############################################







