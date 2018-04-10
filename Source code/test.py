import os
import cv2
import random
import numpy as np
import pandas as pd

from model import *
from losses import *

from cv2 import imread, imwrite
import glob

def get_pair(img_path, mask_path):
    img = imread(img_path, 0)
    img = np.reshape(img,(1,224,224,1))
    img = img/255.0-0.5
    mask = imread(mask_path, 0)
    mask = mask/255.0
    return img, mask


def evaluate(batch_size = 12):
    # load the model
    model = unet_224()
    pretrained_weights_path = 'unet_224_temp.h5'
    model.load_weights(pretrained_weights_path)

    img_filenames = glob.glob('/home/barskaiy/AxSegmenta/data/patched_data_224x224/test/images/control_raw_*')
    mask_filenames = glob.glob('/home/barskaiy/AxSegmenta/data/patched_data_224x224/test/masks/control_raw_*')
    paired_filenames = zip(img_filenames, mask_filenames)
    np.random.shuffle(paired_filenames)
    img_filenames, mask_filenames = zip(*paired_filenames)
    dice_coef_list = []

    for i in range(batch_size):
        idx = np.random.randint(0, len(img_filenames))
        img_path = img_filenames[idx]
        mask_path = mask_filenames[idx]
        img, mask = get_pair(img_path, mask_path)
        img = np.array(img, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)
        mask_predicted = model.predict(img)

        img = np.reshape(img,(224,224,1))
        mask_predicted = np.reshape(mask_predicted, (224,224,1))

        imwrite('show_results/img_%d.png' %i,(img+0.5)*255.0)
        imwrite('show_results/mask_%d.png' %i,mask*255.0)
        imwrite('show_results/mask_predicted_%d.png' %i,mask_predicted*255.0)
        
        dice_coef_list.append(dice_coef(mask,mask_predicted))

    np.savetxt("show_results/dice_coef.csv", dice_coef_list, delimiter=",", fmt='%s')

if __name__ == '__main__':
    evaluate()
