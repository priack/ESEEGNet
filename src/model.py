import tensorflow as tf
from keras.layers import Conv2D, Dense, Dropout, BatchNormalization, Activation, Concatenate, DepthwiseConv2D, Flatten
from keras import Model, Input


def convolution_block(inputs, nFilters: int, size: tuple[int, int], stride: int,  dropRate: float, name: str):
    x = Conv2D(nFilters, size, (1, stride), name=name)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropRate)(x)
    return x


def depth_convolution_block(inputs, length: int, stride: int, dropRate: float, name: str):
    x = DepthwiseConv2D((1, length), (1, stride), use_bias=False, name=name)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropRate)(x)
    return x


def time_convolution_stack(inputs, length: int, nLayers: int, dropRate: float, name: str):
    x = inputs
    for i in range(nLayers):
        x = depth_convolution_block(x, length, length, dropRate, name+str(i))
    # This only works if the final shape is [...., 1, x], otherwise use tf.reduce_mean(x, -2)
    x = tf.squeeze(x, -2)
    return x


def ESEEGNet(shape: tuple) -> Model:
    """
    Assumptions: Last dimmension 5 s windows @ 64 Hz
    :param shape:
    :return:
    """
    dropRate = 0.4
    nTimeFilters = 8
    nSpaceFilters = 6
    nDownTimeFilters = 6
    nFeatures = 14
    batch, d1, d2, t = shape
    # Shape [Batch, d1, d2, t]
    inputs = Input(shape=shape)
    # We unroll the 3D matrix into a 2D to be able to apply DepthWiseConv2D
    # Shape [Batch, d1 * d2, t, 1]
    inputs = tf.reshape(inputs, (batch, d1 * d2, t, 1))

    # We apply several temporal filters to the signal with three different lengths: 0.25 s, 0.5 s, 1s. Then the
    # receptive field is increased with a concatenation of Convolutions with stride.
    # Shape (after first step) [Batch, d1 * d2, t', nTimeFilters]
    # Shape (after second step) [Batch, d1 * d2, nTimeFilters]
    xShort = convolution_block(inputs, nTimeFilters, (1, 16), 8, dropRate, 'ShortTimeFilters')
    xShort = time_convolution_stack(xShort, 6, 2, dropRate, 'ShortTimeReduction')
    xMedium = convolution_block(inputs, nTimeFilters, (1, 32), 16, dropRate, 'MediumTimeFilters')
    xMedium = time_convolution_stack(xMedium, 4, 2, dropRate, 'MediumTimeReduction')
    xLong = convolution_block(inputs, nTimeFilters, (1, 64), 32, dropRate, 'LongTimeFilters')
    xLong = time_convolution_stack(xLong, 3, 2, dropRate, 'LongTimeReduction')
    # Shape [Batch, d1 * d2, nTimeFilters * 3]
    x = Concatenate(axis=-1)([xShort, xMedium, xLong])
    # Shape [Batch, d1, d2, nTimeFilters * 3]
    x = tf.reshape(x, (batch, d1, d2, nTimeFilters * 3))
    # Shape [Batch, d1, d2, nDownTimeFilters]
    x = Conv2D(nDownTimeFilters, (1, 1), name='SpaceReduction')(x)
    xSmall = convolution_block(x, nSpaceFilters, 3, 1, dropRate, 'SmallSizeFilters')
    xSmall = Flatten(name='SmallFeatures')(xSmall)
    # Shape [Batch, features]
    # feats = Concatenate(axis=-1)([xSmall, xMid, xLarge])
    hidden = Dense(nFeatures, activation='relu', name='Hidden')(xSmall)
    output = Dense(2, activation='softmax', name='Softmax')(hidden)

    mdl = Model(inputs=inputs, outputs=output)
    mdl.summary()
    return mdl

shape = (100, 8, 7, 321)
mdl = ESEEGNet(shape)
