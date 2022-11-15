from keras.models import Model
from keras.layers import (Input, Conv2D, MaxPooling2D,
                          concatenate, Conv2DTranspose, Dropout)


def unet_conv_block(x, filters=16, kernel_size=(2, 2),
                    strides=1,
                    # pool=None,
                    dropout=0.1):

    # 1st conv2d
    x = Conv2D(filters, kernel_size, activation='relu',
               padding='same', strides=strides)(x)

    # dropout
    x = Dropout(dropout)(x)

    # second conv layer
    x = Conv2D(filters, kernel_size, activation='relu',
               padding='same', strides=strides)(x)

    return x


def simple_unet_model(**kwargs):
    # Build the model
    # Model designed for input shape of 12, 12, 1
    inputs = Input(kwargs["img_shape"])  # 12x12x1

    # stage 1 - contract
    c1 = unet_conv_block(inputs, 16, 2)  # 12x12x16
    p1 = MaxPooling2D(2)(c1)  # 6x6x16

    # stage 2 - contract
    c2 = unet_conv_block(p1, 32, 2)  # 6x6x16
    p2 = MaxPooling2D(2)(c2)  # 3x3x16

    # center
    H = unet_conv_block(p2, 64, 2)  # 3x3x64

    # stage 2 - expand
    e2 = Conv2DTranspose(32, 2, strides=2, padding='same')(H)  # 6x6x32
    e2 = concatenate([e2, c2])  # 6x6x64
    e2 = unet_conv_block(e2, 16, 2)  # 6x6x16

    # stage 1 - expand
    e1 = Conv2DTranspose(16, 2, strides=2, padding='same')(e2)  # 12x12x16
    e1 = concatenate([e1, c1])  # 12x12x32
    e1 = unet_conv_block(e1, 16, 2)  # 12x12x16

    outputs = Conv2D(1, (1, 1), activation='sigmoid', name="output")(e1)

    model = Model(inputs=[inputs], outputs=[outputs])

    # Do not compile here
    if kwargs.get("debug", False):
        model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['accuracy'])
        model.summary()

    return model


if __name__ == "__main__":
    simple_unet_model(img_shape=(12, 12, 1), debug=True)
