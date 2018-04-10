import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
from keras import backend as K

import glob
from scipy.io import loadmat
import cv2
from cv2 import resize
from cv2 import imwrite
import h5py

# set up datapathes
brain_control_list = ['MB2b', 'MB2d', 'MB3a', 'MB5a', 'MB6d', 'MB7c']
basic_path = '../../../data/west/Data/Mouse_brains/Histology/From_computer/results/'
imgs_data_path = []

for brain in brain_control_list:
    data_path = basic_path + brain + '/'
    brain_paths = glob.glob(data_path+'*/*_thresh_grow_myth_equal.mat')
    for path in brain_paths:
        imgs_data_path.append(path)

# set up parameters of images
width = 2048
height = 2048
num_examples = len(imgs_data_path)

# load each .mat file, from .mat file extract an image and a mask
for i, img_file in enumerate(imgs_data_path):
    print('Loading {} file'.format(imgs_data_path[i]))
    mat_structure = loadmat(img_file)['axmy'][0,0]
    img = mat_structure['img']
    mask = mat_structure['BW_myelin']

    img = np.array(img.astype('float32'))
    mask = np.array(mask.astype('float32'))
    if img.shape != (width, height):
        img = resize(img, (width, height))
    if mask.shape != (width, height):
        mask = resize(mask, (width, height))

    # save each image and each mask
    imwrite('data/raw_data/images/control_raw_%d.png' %i,img)
    imwrite('data/raw_data/masks/control_raw_%d.png' %i,mask*255)

    # for each image make patches
    patch_width = 224
    patch_height = 224
    patches_num = 100

    img_patches = np.zeros((patches_num * num_examples,
                            patch_width, patch_height))
    masks_patches = np.zeros((patches_num * num_examples,
                              patch_width, patch_height))

    for j in range(patches_num):
            rand_x = np.random.randint(0, width - patch_width +1)
            rand_y = np.random.randint(0, height - patch_height +1)
            top = rand_y
            left = rand_x
            bottom = top + patch_height
            right = left + patch_width
            img_patch = img[top:bottom, left:right]
            mask_patch = mask[top:bottom, left:right]
            number = i*patches_num + j

            imwrite('data/patched_data_224x224/images/control_raw_%d.png' %number,
                    img_patch)
            imwrite('data/patched_data_224x224/masks/control_raw_%d.png' %number,
                    mask_patch*255)

            img_patches[number] = img_patch
            masks_patches[number] = mask_patch

# save n-dimentional array into hdf5 file
hdf5_path = 'data/patched_data_224x224/control_hdf5/control_data.h5'
data_shape = (len(img_patches), patch_height, patch_width)

# open a hdf5 file and create arrays
hdf5_file = h5py.File(hdf5_path, mode='w')
hdf5_file.create_dataset("data", data_shape, np.float64)
hdf5_file.create_dataset("masks", data_shape, np.float64)

# loop over images
for i in range(len(data_shape)):
    if i % 1000 == 0 and i > 1:
        print 'Train data: {}/{}'.format(i, len(data_shape))
        hdf5_file["data"][i, ...] = img_patches[i]
        hdf5_file["masks"][i, ...] = masks_patches[i]

hdf5_file.close()
