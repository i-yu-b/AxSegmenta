from keras import backend as K

def dice_coef(y_true, y_pred):
    bias = 1.0 # to set dice coefficient = 1, when sets are empty
    # flatten masks
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersection = K.sum(y_true_flat * y_pred_flat)
    dice_coef = (2. * intersection + bias) / (
                 K.sum(y_true_flat) + K.sum(y_pred_flat) + bias)
    return dice_coef

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)