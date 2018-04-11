import os
import cv2
import random
import numpy as np
import pandas as pd

from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import __version__

from model import *
from losses import *
from data_generation import *

def train_unet():
    out_model_path = '/notebooks/b.irina/AxSegmenta/checkpoints/unet_224.h5'
    epochs = 400
    patience = 20
    batch_size = 12
    learning_rate = 0.001
    model = unet_224()

    # load from pretrained model
   # pretrained_weights_path = 'zf_unet_224.h5'
    #if os.path.isfile(pretrained_weights_path):
   #     model.load_weights(pretrained_weights_path)

    #optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    optim = Adam(lr=learning_rate)
    model.compile(optimizer=optim, loss=dice_coef_loss, metrics=[dice_coef])
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        ModelCheckpoint('unet_224_temp.h5', monitor='val_loss', save_best_only=True, verbose=0),
    ]

    print('Start training...')
    history = model.fit_generator(
        generator=batch_generator(batch_size, 1),
        epochs=epochs,
        steps_per_epoch=100,
        validation_data=batch_generator(batch_size, 0),
        validation_steps=100,
        verbose=2,
        callbacks=callbacks)

    model.save_weights(out_model_path)
    pd.DataFrame(history.history).to_csv('unet_224_train.csv', index=False)
    print('Training is finished (weights unet_224.h5 and log unet_224_train.csv are generated )...')

if __name__ == '__main__':
    try:
        from tensorflow import __version__ as __tensorflow_version__
        print('Tensorflow version: {}'.format(__tensorflow_version__))
    except:
        print('Tensorflow is unavailable...')

    print('Keras version {}'.format(__version__))

    train_unet()
