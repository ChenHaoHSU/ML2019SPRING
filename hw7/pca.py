import os
import sys
import numpy as np 
from skimage.io import imread, imsave

image_dir = sys.argv[1]
input_fpath = sys.argv[2]
output_fpath = sys.argv[3]
print('# [Info] Argv')
print('    - Image directory     : {}'.format(image_dir))
print('    - Input image         : {}'.format(input_fpath))
print('    - Reconstructed image : {}'.format(output_fpath))

# Number of principal components used
k = 5

def process(M): 
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M

def load_images(fpath):
    filelist = os.listdir(fpath) 
    img_data = []
    for i in range(415):
        filename = os.path.join(fpath, '{}.jpg'.format(i))
        print('\r# [Info] Loading image: {}'.format(filename), end='', flush=True)
        tmp = imread(filename)
        img_shape = tmp.shape
        img_data.append(tmp.flatten())
    print('', flush=True)
    return np.array(img_data).astype('float32'), img_shape

# Load all images
image_data, img_shape = load_images(image_dir)

# Calculate mean & Normalize
mean = np.mean(image_data, axis = 0)
image_data -= mean 

# Use SVD to find the eigenvectors 
print('# [Info] Decomposition...')
u, s, v = np.linalg.svd(image_data.T, full_matrices=False)
print('    - u.shape: {}'.format(u.shape))
print('    - s.shape: {}'.format(s.shape))
print('    - v.shape: {}'.format(v.shape))

################
# Reproduce
################

# Load image & Normalize
filename = os.path.join(image_dir, input_fpath)
print('# [Info] Reproduce: {}'.format(filename))
picked_img = imread(filename)
X = picked_img.flatten().astype('float32')
X -= mean
# Compression
weights = np.dot(X, u[:, :k])
# Reconstruction
M = np.dot(weights, u[:, :k].T) + mean
reconstruct = process(M)
imsave(output_fpath, reconstruct.reshape(img_shape))
print('    - Save {}!'.format(output_fpath))

################
# Report
################

# Report Problem 1.c
print('# [Info] Problem 1.c')
test_image = ['1.jpg','10.jpg','22.jpg','37.jpg','72.jpg']
for x in test_image:
    # Load image & Normalize
    filename = os.path.join(image_dir, x)
    print('    - Load {}...'.format(filename))
    picked_img = imread(filename)
    X = picked_img.flatten().astype('float32')
    X -= mean

    # Compression
    weights = np.dot(X, u[:, :k])
    
    # Reconstruction
    M = np.dot(weights, u[:, :k].T) + mean
    reconstruct = process(M)
    filename_rec = x[:-4] + '_reconstruction.jpg'
    imsave(filename_rec, reconstruct.reshape(img_shape))
    print('    - Save {}!'.format(filename_rec))

# Report Problem 1.a
print('# [Info] Problem 1.a')
average = process(mean)
filename = 'mean.jpg'
imsave(filename, average.reshape(img_shape))  
print('    - Save {}!'.format(filename))

# Report Problem 1.b
print('# [Info] Problem 1.b')
for i in range(10):
    eigenface = process(u[:, i].T)
    filename = '{}_eigenface.jpg'.format(i)
    imsave(filename, eigenface.reshape(img_shape)) 
    print('    - Save {}!'.format(filename))

# Report Problem 1.d
print('# [Info] Problem 1.d')
for i in range(5):
    number = s[i] * 100 / sum(s)
    print('    - {} - {:<2.2f}%'.format(i, number))
