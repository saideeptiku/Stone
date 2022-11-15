"""
An implementation of Siamese Neural Network using Triplet Loss
"""
# supress warning msgs
# I know I don not have a GPU!
# https://stackoverflow.com/a/42121886
#yapf: disable
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#yapf: enable

import json
from scipy.stats import truncnorm
from tensorflow import keras
import tensorflow as tf
import numpy as np
from scipy.stats import multivariate_normal

if __name__ == "__main__":
    from uji import UJI


class TripletManager(keras.utils.Sequence):
    def __init__(
            self,
            x,
            y,
            n_sampler,
            bs=32,
            steps_per_epoch=32,
            # variables associated with effects
            p_turn_off=None,  # 0 - 1
            contrast_range=None,  # (low=0.5, high)
            brightness_delta=None,  # 0 - 1
    ):
        """
        Init for Triplet Manager
        if path mat is provided; a separate method for negative selection is used
        otherwise, a 1d path is assumed

        Args:
        ----
        x: input features
        y: output features
        n_sampler: should be a function, that given Anchor label and all possible labels; 
                   provides Negative label; looks like n_sampler(anchor: int, labels: list)
        """
        self.x, self.y = x, y
        self.labels = np.unique(y.flatten())

        # make y 2D
        if len(y.shape) == 1:
            self.y = self.y.reshape((-1, y.shape[0]))

        self.bs = bs
        self.steps_per_epoch = steps_per_epoch
        self.n_sampler = n_sampler

        # variables associated with effects
        self.p_turn_off = p_turn_off
        self.contrast_range = contrast_range
        self.brightness_delta = brightness_delta

    def get_sample(self, label, batch_size=32):

        mask = self.y == label
        mask = mask.flatten()

        x_label = self.x[mask, :]

        # pick random samples from x_stack for batch size
        inds = np.random.choice(x_label.shape[0], size=batch_size)

        return x_label[inds, :]

    def pick_sample(self, batch_size=32):
        """
        pick three difficult samples
        from analysis we know samples closeby are difficult
        we use this information through a probability distribution
        """
        # Anchor and Positive Label
        ap_label = np.random.choice(self.labels)
        # negative label
        n_label = self.n_sampler(ap_label, self.labels)

        # make two sets of samples
        anchor = self.get_sample(ap_label, batch_size=batch_size)
        # turn of APs at random
        # if self.p_turn_off is not None:
        #     anchor = self.apply_turn_off(anchor)

        # get +ve and apply effects
        positive = self.get_sample(ap_label, batch_size=batch_size)
        positive = self.apply_effects(positive)

        # get -ve and apply effects
        negative = self.get_sample(n_label, batch_size=batch_size)
        negative = self.apply_effects(negative)

        # dummy label, not really used
        labels = np.ones(batch_size)

        return [anchor, positive, negative], labels

    def apply_effects(self, tx):
        """Apply effects as described in constructor

        Parameters
        ----------
        tx : np.array
            a sample or set of samples; expects 2D
        """
        for i, row in enumerate(tx):

            # apply effect if provided
            if self.p_turn_off is not None:
                row = random_turn_off(row, self.p_turn_off)
                # save the row
                tx[i, :] = row

        if self.contrast_range is not None:
            tx = random_contrast(tx, *self.contrast_range)
        if self.brightness_delta is not None:
            tx = random_brightness(tx, self.brightness_delta)

        return tx

        pass

    def gen_samples(self):
        """
        Generator for this class
        """
        while True:
            yield self.pick_sample(batch_size=self.bs)

    # function for keras.Models.Sequence that are actually called

    def on_epoch_end(self):
        """ ehh... """
        pass

    def __len__(self):
        """
        """
        return self.steps_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        return self.pick_sample(batch_size=self.bs)


###########################################
# Sample Utility functions
# Can be combined to be used as n_sampler method for linear paths
###########################################


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def gen_pdf(mean, sd, low, upp, normed=False):
    x = get_truncated_normal(mean=mean, sd=sd, low=low, upp=upp)
    x = x.rvs(1000)
    hist, bin_edges = np.histogram(x, bins=upp + 1)

    if normed:
        return hist / np.sum(hist)

    return hist


def pick_negative_1d(anchor: int, labels, sd=4):
    """
    Given the anchor label and all known labels
    Can be used as an n_sampler method
    """

    dist = gen_pdf(anchor, sd, np.min(labels), np.max(labels))

    # remove probabitiy of getting same label
    dist[anchor] = 0.0

    # normalize
    dist = dist / np.sum(dist)

    # given the probability distribution pick a label
    return np.random.choice(np.sort(labels), p=dist)


###########################################
# Functions used to train the Triplet Loss Encoder
###########################################


def identity_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)


def triplet_loss(x, alpha=0.2):
    # Triplet Loss function.
    anchor, positive, negative = x
    # distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    # distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.maximum(basic_loss, 0.0)
    return loss


def embedding_model(
        input_shape: tuple,
        num_outputs: int,
        gaussian_noise=None,
        custom_layers=None,
        model_layers=[50, 50, 100],
        kernel_size=(2, 2),
        strides=1,
):
    """A simple con2d based encoder
    if a custom layers are passed, 
    only input and output layers are added

    Parameters
    ----------
    input_shape : tuple
        input image shape
    num_outputs : int
        number of outputs, in the case size of the embedding
    custom_layers : Iterable, optional
        if a list of custom layers are provided, 'model_layers' is ignored, by default None
    model_layers : list, optional
        filtters in layers, by default [50, 50, 100]
    kernel_size : tuple, optional
        common ks in Conv2D layers, by default (2, 2)
    strides : int, optional
        Conv2D strides, by default 1
    gaussian_noise : float, optional
        gaussian noise added to input, by default None

    Returns
    -------
    tf.Model
        Embedding model created in this function
    """

    # make input layer
    input_layer = keras.Input(shape=input_shape)
    conn = input_layer

    if custom_layers is None:

        # add gaussian noise if provided
        if gaussian_noise is not None:
            conn = keras.layers.GaussianNoise(gaussian_noise)(conn)

        # add conv body layers
        for i, num_filter in enumerate(model_layers[:-1]):

            # include convolution layers
            conn = keras.layers.Conv2D(
                num_filter,
                kernel_size,
                strides=strides,
                activation='relu',
            )(conn)
            conn = keras.layers.Dropout(0.1)(conn)

        # flatten layers
        conn = keras.layers.Flatten()(conn)

        # flattened layers post conv
        conn = keras.layers.Dense(
            model_layers[-1],
            activation='relu',
        )(conn)
        conn = keras.layers.Dropout(0.2)(conn)

    else:
        for my_layer in custom_layers:
            conn = my_layer(conn)

    # output layer
    output_layer = keras.layers.Dense(num_outputs)(conn)

    return keras.Model(input_layer, output_layer)


def freezer(model):
    for layer in model.layers:
        layer.trainable = False


def thaw(model):
    for layer in model.layers:
        layer.trainable = True


def is_frozen(model):
    for layer in model.layers:
        if layer.trainable:
            return False
    return True


def complete_model(base_model, input_shape, lr, alpha):
    # Create the complete model with three
    # embedding models and minimize the loss
    # between their output embeddings
    input_1 = keras.layers.Input(input_shape)
    input_2 = keras.layers.Input(input_shape)
    input_3 = keras.layers.Input(input_shape)

    A = base_model(input_1)
    P = base_model(input_2)
    N = base_model(input_3)

    def trip_loss(x):
        return triplet_loss(x, alpha)

    # ooh magic!
    loss = keras.layers.Lambda(trip_loss)([A, P, N])
    model = keras.Model(inputs=[input_1, input_2, input_3], outputs=loss)
    model.compile(loss=identity_loss, optimizer=keras.optimizers.Adam(lr))
    return model


def random_turn_off(img, p, value_range=(0, 1)):
    """randomly turn on some percentage of pixels
    that were previously invisible

    :param img: numpy array that can be interpreted as an image
    :param p: percentage value -- between 0 and 1
    :param value_range: range between with all data exists
    :returns: random contrast array
    :rtype: numpy
    """

    # save shape
    img_shape = img.shape
    img = img.flatten()

    # number of pixels that are on
    (visible_indices, ) = np.where(img > value_range[0])
    num_visible_pixels = visible_indices.shape[0]

    # from uniform distribution
    p_off = np.random.uniform(low=0, high=p)
    num_set_off = int(p_off * num_visible_pixels)

    # percentage is smaller than 1 pixel
    if num_set_off == 0:
        return img.reshape(img_shape)

    # indices where pixels are set off
    ind_set_off = np.random.choice(visible_indices, size=num_set_off, replace=False)

    # turn of pixels
    img[ind_set_off] = value_range[0]

    # reshape and return
    return img.reshape(img_shape)


def random_contrast(vector, min_val, max_val, value_range=(0, 1)):
    """apply tensorflow random contrast with a mask
    contrast is only applied on pixels that are visible

    :param img: numpy array that can be interpreted as an image
    :param min_val: min value for random contrast
    :param max_val: max value for random contrast
    :param value_range: range between with all data exists
    :returns: random contrast array
    :rtype: numpy

    """

    # create a mask
    # store all places where value was originally zero
    mask = vector == 0

    # apply random contrast; turn back into numpy array
    # print(vector.shape)
    # vector = vector.reshape((1, 1, vector.shape[0], 1))
    vector = np.array(tf.image.random_contrast(vector, min_val, max_val))
    # vector = vector.flatten()

    # restore zero in places that was saved before as mask
    vector[mask] = 0

    # clip is needed such that any new values that
    # are negative should become zero again
    return np.clip(vector, *value_range)


def random_brightness(vector, max_delta, value_range=(0, 1)):
    """apply tensorflow random brightness with a mask
    brightness is only applied on pixels that are visible

    :param img: numpy array that can be interpreted as an image
    :param min_val: min value for random contrast
    :param max_val: max value for random contrast
    :param value_range: range between with all data exists
    :returns: random contrast array
    :rtype: numpy

    """
    # create a mask
    # store all places where value was originally zero
    mask = vector == 0
    # print(vector.shape)

    # apply random contrast; turn back into numpy array
    # vector = vector.reshape((1, 1, vector.shape[0], 1))
    vector = np.array(tf.image.random_brightness(vector, max_delta))
    # vector = vector.flatten()

    # restore zero in places that was saved before as mask
    vector[mask] = 0

    # clip is needed such that any new values that
    # are negative should become zero again
    return np.clip(vector, *value_range)


def pick_negative_uji(anchor: int, labels: list, sigma=30):

    num_labels = len(set(labels))

    # build cov mat
    sigma = np.array([
        [
            sigma,
            -0,
        ],
        [-0, sigma],
    ])

    # floor mat and label2ij
    floor_mat, label2ij = UJI.uji_floor_mat_and_dict()

    # find the i and j coordinate of label in floor_mat
    i, j = label2ij[anchor]

    # mu should be at anchor
    mu = np.array([i, j])

    # create a multivariate normal
    bi_normal = multivariate_normal(mu, sigma)

    # Note: this might be inverted
    X = np.arange(floor_mat.shape[0])
    Y = np.arange(floor_mat.shape[1])
    X, Y = np.meshgrid(X, Y)
    # pdf calls require this mat
    pos = np.empty(X.shape + (2, ))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    Z = bi_normal.pdf(pos)
    # set probability at label to zero
    Z[j, i] = 0.0

    # get probabilities for each label
    label_probs = np.zeros((num_labels, ))

    for label, (i, j) in label2ij.items():
        try:
            label_probs[label] = Z[j, i]
        except:
            pass

    label_probs = label_probs / np.sum(label_probs)
    return np.random.choice(np.arange(num_labels), p=label_probs)


def save_encoder(encoder: tf.keras.Model, meta: dict, save_path: str):
    encoder.save(save_path)
    json.dump(meta, open(f"{save_path}/model_info.json", "w"))


def load_encoder(save_path: str):
    encoder = tf.keras.models.load_model(save_path, compile=False)
    meta = json.load(open(f"{save_path}/model_info.json", "r"))
    return encoder, meta


if __name__ == "__main__":

    from matplotlib import pyplot as plt

    x = np.random.random((10))

    plt.plot(x)
    y = random_contrast(x, 0.5, 1.5)
    plt.plot(y)

    plt.show()

    exit()
