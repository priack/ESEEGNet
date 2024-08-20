import tensorflow as tf
from keras.layers import Conv2D, Conv3D, Dense, Dropout, MaxPool3D, MaxPool2D, BatchNormalization, Activation, Concatenate, DepthwiseConv2D, AvgPool2D
from keras import Model, Input


def convolution_block(inputs, nFilters: int, length: int, stride: int,  dropRate: float, name: str):
    x = Conv2D(nFilters, (1, length), (1, stride), padding='same', name=name)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropRate)(x)
    return x


def depth_convolution_block(inputs, length: int, stride: int, dropRate: float):
    x = DepthwiseConv2D((1, length), (1, stride))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropRate)(x)
    return x


def time_convolution_stack(inputs, length: int, nLayers: int, dropRate: float):
    x = inputs
    for i in range(nLayers):
        x = depth_convolution_block(x, length, length, dropRate)
    x = tf.reduce_mean(x, -2)
    return x


def ESEEGNet(shape: tuple) -> Model:
    """
    Assumptions: Last dimmension 5 s windows @ 64 Hz
    :param shape:
    :return:
    """
    dropRate = 0.4
    nTimeFilters = 8
    batch, d1, d2, t = shape
    # Shape [Batch, d1, d2, t]
    inputs = Input(shape=shape)
    # We unroll the 3D matrix into a 2D to be able to apply DepthWiseConv2D
    # Shape [Batch, d1 * d2, t, 1]
    inputs = tf.reshape(inputs, (batch, d1 * d2, t, 1))

    # We apply several temporal filters to the signal with three different lengths: 0.25 s, 0.5 s, 1s. Then the
    # receptive field is increased with a concatenation of Convolutions with stride.
    # Shape [Batch, d1 * d2, t', nTimeFilters]
    # Shape [Batch, d1 * d2, nTimeFilters]
    xShort = convolution_block(inputs, nTimeFilters, 16, 8, dropRate, 'ShortTimeFilters')
    xShort = time_convolution_stack(xShort, 6, 2, dropRate)
    xMedium = convolution_block(inputs, nTimeFilters, 32, 16, dropRate, 'MediumTimeFilters')
    xMedium = time_convolution_stack(xMedium, 4, 2, dropRate)
    xLong = convolution_block(inputs, nTimeFilters, 64, 32, dropRate, 'LongTimeFilters')
    xLong = time_convolution_stack(xLong, 3, 2, dropRate)
    # Shape [Batch, d1 * d2, nTimeFilters * 3]
    xERPMap = Concatenate(axis=-1)([xShort, xMedium, xLong])
    xERPMap = tf.reshape(xERPMap, (batch, d1, d2, nTimeFilters * 3))

    mdl = Model(inputs=inputs, outputs=xERPMap)
    mdl.summary()

shape = (100, 8, 7, 321)
mdl = ESEEGNet(shape)

