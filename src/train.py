from model import ESEEGNet
import keras
import numpy as np
import tensorflow as tf
import dill

with open('./data/full_classReady.dill', 'rb') as f:
    data = dill.load(f)
batchSize = 128
x = data['x']
y = data['y']
nTrials, rows, col, t = x.shape
shape = (rows, col, t)
mdl = ESEEGNet(shape)
mdl.compile(
    loss=keras.losses.binary_crossentropy,
    optimizer=keras.optimizers.Adam(0.001),
    metrics=[keras.metrics.binary_crossentropy, keras.metrics.F1Score],
)

trainLen = int(nTrials * 0.8)
idx = np.random.permutation(nTrials)
x = x[idx]
y = y[idx, 0]
xtr, ytr = x[:trainLen], y[:trainLen]
xtr = np.reshape(xtr, (trainLen, rows * col, t, 1))
ytr = tf.one_hot(ytr, 2)
mdl.fit(xtr, ytr, batch_size=batchSize, epochs=100)