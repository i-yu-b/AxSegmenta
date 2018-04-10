import numpy as np
import pandas as pd
import os
import random
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
    axPos = mat_structure['axPos']
    axPos_edge = mat_structure['axPos_edge']


    axPos = np.array(axPos.astype('float32'))
    axPos_edge = np.array(axPos_edge.astype('float32'))

    np.savetxt("data/raw_data/axon_position/axon_%d.csv", axPos, delimiter=",")
    np.savetxt("data/raw_data/axon_position/axon_edge_%d.csv", axPos_edge, delimiter=",")
