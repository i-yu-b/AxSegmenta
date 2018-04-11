'''
    Custom simple data generator
'''

import numpy as np
from cv2 import imread
import glob

def get_pair(img_path, mask_path):
    img = imread(img_path, 0)
    img = np.reshape(img,(224,224,1))
    img = img/255.0-0.5
    mask = imread(mask_path, 0)
    mask = mask/255.0
    return img, mask

def batch_generator(batch_size, train = True):
    if train:
        img_filenames = glob.glob('/notebooks/b.irina/AxSegmenta/Data/train/images/control_raw_*')
        mask_filenames = glob.glob('/notebooks/b.irina/AxSegmenta/Data/train/masks/control_raw_*')
    else:
        img_filenames = glob.glob('/notebooks/b.irina/AxSegmenta/Data/valid/images/control_raw_*')
        mask_filenames = glob.glob('/notebooks/b.irina/AxSegmenta/Data/valid/masks/control_raw_*')

    paired_filenames = list(zip(img_filenames, mask_filenames))
    np.random.shuffle(paired_filenames)
    img_filenames, mask_filenames = zip(*paired_filenames)

    while True:
        image_list = []
        mask_list = []
        for i in range(batch_size):
            idx = np.random.randint(0, len(img_filenames))
            img_path = img_filenames[idx]
            mask_path = mask_filenames[idx]
            img, mask = get_pair(img_path, mask_path)
            image_list.append(img)
            mask_list.append([mask])
        image_list = np.array(image_list, dtype=np.float32)
        mask_list = np.array(mask_list, dtype=np.float32)

        yield image_list, mask_list
