import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="-1 "
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from keras.optimizers import Adam, SGD
from model import get_unet
from losses import dice_coef, dice_coef_loss
from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator
import h5py
from keras.callbacks import ModelCheckpoint
import tensorflow as tf


def train_simple():
    # load data
    hdf5_path = 'data/patched_data/control_hdf5/control_data.h5'
    hdf5_file = h5py.File(hdf5_path, "r")
    data = hdf5_file["data"][...]
    masks = hdf5_file["masks"][...]
    data = data.reshape(-1,512,512,1)
    masks = masks.reshape(-1,512,512,1)

    # set up learning parameters
    learning_rate = 1e-5
    epochs = 40
    batch_size = 32

    # load the model
    model = get_unet(input_shape=(512, 512, 1))
    model.summary()
    model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss,
                  metrics=[dice_coef])

    checkpoints_folder = 'saved_model/'
    checkpointer = ModelCheckpoint(
        checkpoints_folder + '/ep{epoch:02d}-vl{val_loss:.4f}.hdf5',
        monitor='loss')

    model.fit(data, masks, batch_size = batch_size,
                    epochs = epochs,
                    callbacks=[checkpointer],
                    verbose=1, shuffle=True,
                               validation_split=0.2)

if __name__ == '__main__':
    train_simple()
