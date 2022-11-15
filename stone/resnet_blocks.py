"""
Implementational details and references

[1] Medium Article, https://towardsdatascience.com/understand-and-implement-resnet-50-with-tensorflow-2-0-1190b9b52691
[2] ResNet Paper, https://arxiv.org/pdf/1512.03385.pdf
"""

import tensorflow as tf


class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters, scope=''):
        super(ResnetIdentityBlock, self).__init__(name=scope)
        filters1, filters2 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters1, kernel_size,
                                             padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters2, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)


class ResnetConvolutionBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters, stride=2, scope=''):
        super(ResnetConvolutionBlock, self).__init__(name=scope)
        filters1, filters2 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1,
                                             (1, 1),
                                             strides=stride)
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters1, kernel_size,
                                             padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters2, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

        # shortcut
        self.conv2d = tf.keras.layers.Conv2D(filters2,
                                             (1, 1),
                                             strides=stride)
        self.bn2d = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)
        # no act

        x_short = self.conv2d(input_tensor)
        x_short = self.bn2d(x, training=training)
        # no act short

        # combine
        x = x + x_short

        return tf.nn.relu(x)


if __name__ == "__main__":

    # define test fucntions
    def test_build_resnet():

        import numpy as np
        from tensorflow import keras
        from tensorflow.keras import (
            losses,
            optimizers,
            layers
        )

        x = np.random.random((64, 10, 10, 1))
        y = np.random.randint(0, 19, size=(64, 1))

        blocks = [
            # block 1
            # ResnetIdentityBlock(3, [32, 64]),
            ResnetConvolutionBlock(3, [32, 64]),
            ResnetIdentityBlock(3, [32, 64]),
            ResnetIdentityBlock(3, [32, 64]),

            # block 2
            ResnetConvolutionBlock(3, [64, 128]),
            ResnetIdentityBlock(3, [64, 128]),
            ResnetIdentityBlock(3, [64, 128]),

            # block 3
            ResnetConvolutionBlock(3, [128, 256]),
            ResnetIdentityBlock(3, [128, 256]),
            ResnetIdentityBlock(3, [128, 256]),

        ]

        model = keras.models.Sequential([
            layers.Input(shape=x.shape[1:]),
            *blocks,
            layers.Flatten(),
            layers.Dense(20, activation='softmax')
        ])

        model.compile(loss=losses.sparse_categorical_crossentropy,
                      optimizer=optimizers.Adam())

        print(model.summary())
        model.fit(x, y)

    # test_resnet_bottleneck()
    test_build_resnet()
