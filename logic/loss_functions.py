# %%
import numpy as np
from tensorflow.keras import backend
import tensorflow as tf

# Loss function defined as Dice Coeffiecient
smooth = 1.0

def dice_coef(y_true, y_pred):
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + smooth)

def dice_coef_negative(y_true, y_pred, smooth = 1.0):
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    y_true_neg_f = 1 - y_true_f
    y_pred_neg_f = 1 - y_pred_f
    intersection_neg = backend.sum(y_true_neg_f * y_pred_neg_f)
    return ((2. * intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + smooth)) - ((2. * intersection_neg + smooth) / (backend.sum(y_true_neg_f) + backend.sum(y_pred_neg_f) + smooth))

def dice_coef_loss(y_true, y_pred):
    return 2 - dice_coef(y_true, y_pred)


def dice_coeff_standard(y_true, y_pred, thresold = 0.5):
    y_true = y_true > thresold
    y_pred = y_pred > thresold
    # print(y_pred)
    # print(y_true)
    sum = np.sum(y_true) + np.sum(y_pred)
    # print(sum)
    andResult = np.logical_and(y_true, y_pred)
    # print(andResult)
    sumAndResult = np.sum(andResult)
    # print(sumAndResult)
    return 2 * sumAndResult / sum

def segmentation_overlap_standard(y_true, y_pred, thresold = 0.5):
    y_true = y_true > thresold
    y_pred = y_pred > thresold
    total = np.sum(y_true)
    intersection = np.sum(np.logical_and(y_true, y_pred))
    return intersection / total

def binary_accuracy_standard(y_true, y_pred, thresold = 0.5):
    y_true = y_true > thresold
    y_pred = y_pred > thresold
    nxor = np.logical_not(np.logical_xor(y_true, y_pred))
    total = np.sum(nxor)
    return total / y_true.size
    
def segmentation_overlap(y_true, y_pred):
    # y_true_f = backend.flatten(y_true)
    # y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true * y_pred)
    total = backend.sum(y_true)
    epsilon = backend.epsilon() # to avoid NaN's and Inf's
    return (intersection + epsilon) / (total + epsilon) # if there's nothing to segment, then we should consider 100%

def combined_dice_coef_and_overlap_loss(y_true, y_pred):
    intersection = backend.sum(y_true * y_pred)
    total = backend.sum(y_true)
    total_predicted = backend.sum(y_pred)
    epsilon = backend.epsilon() # to avoid NaN's and Inf's
    return 3 - (2. * intersection + smooth) / (total + total_predicted + smooth) + intersection / (total + epsilon)


def segmentation_overlap_loss(y_true, y_pred):
     return 1 - segmentation_overlap(y_true, y_pred)

# Focal loss https://arxiv.org/abs/1708.02002
def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        # y_true_f = backend.flatten(y_true)
        # y_pred_f = backend.flatten(y_pred)
        # pt_1 = tf.where(tf.equal(y_true_f, 1), y_pred_f, tf.ones_like(y_pred_f))
        # pt_0 = tf.where(tf.equal(y_true_f, 0), y_pred_f, tf.zeros_like(y_pred_f))

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = backend.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = backend.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = backend.clip(pt_0, epsilon, 1. - epsilon)

        return -backend.mean(alpha * backend.pow(1. - pt_1, gamma) * backend.log(pt_1)) \
               -backend.mean((1 - alpha) * backend.pow(pt_0, gamma) * backend.log(1. - pt_0))

    return binary_focal_loss_fixed

def combined_focal_and_overlap_loss(y_true, y_pred):
    def binary_focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = backend.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = backend.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = backend.clip(pt_0, epsilon, 1. - epsilon)

        return -backend.mean(0.25 * backend.pow(1. - pt_1, 2.) * backend.log(pt_1)) \
            -backend.mean((1 - 0.25) * backend.pow(pt_0, 2.) * backend.log(1. - pt_0))
    return binary_focal_loss_fixed(y_true, y_pred) + segmentation_overlap_loss(y_true, y_pred)

#%%
