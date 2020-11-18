import matplotlib.pyplot as plt
import tensorflow
import numpy as np
from tensorflow import nn as nn

def flatten(layer):
    shape = layer.get_shape()
    nFeatures = shape[1:4].num_elements()
    
    flatten = tensorflow.reshape(layer, [-1, nFeatures])
    return flatten, nFeatures

def convolution_layer(data, nChannels, filter_size, nFilters):
    kernelShape = [filter_size, filter_size, nChannels, nFilters]
    weights = tensorflow.Variable(tensorflow.truncated_normal(kernelShape, stddev=0.05))
    biases = tensorflow.Variable(tensorflow.constant(0.05, shape=[nFilters]))
    
    layer = nn.conv2d(data, weights, [1, 1, 1, 1], 'SAME')
    layer = nn.bias_add(layer, biases)
    layer = nn.max_pool(layer, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    layer = nn.relu(layer)
    
    return layer

def fully_connected(data, nInput, nOutput):
    weights = tensorflow.Variable(tensorflow.truncated_normal([nInput, nOutput], stddev=0.05))
    biases = tensorflow.Variable(tensorflow.constant(0.05, shape=[nOutput]))

    layer = tensorflow.matmul(data, weights)
    layer = nn.bias_add(layer, biases)

    layer = nn.relu(layer)

    return layer

def choose_optimizer(optimizer, cost):
    if optimizer == "Adam":
        return tensorflow.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
    
    elif optimizer == "RMS":
        return tensorflow.train.RMSPropOptimizer(learning_rate=1e-4).minimize(cost)

def run_optimizer(n, data, optimizer, session, x, y_true):
    for i in range(n):
        x_value, y_value = data.train.next_batch(256)
        dictionary = {x: x_value, y_true: y_value}
        session.run(optimizer, feed_dict=dictionary)

def get_accuracy(data, session, x, y_true, y_pred_cls):
    d = data.test
    n = len(d.images)
    layer_pred = np.zeros(shape=n, dtype=np.int)
    i = 0
    while i < n:
        j = min(i + 256, n)
        images = d.images[i:j, :]
        labels = d.labels[i:j, :]
        feed_dict = {x: images, y_true: labels}
        layer_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j
    d.conv = np.argmax(data.test.labels, axis=1)
    true = d.conv
    correct = (true == layer_pred)
    correct_sum = correct.sum()
    accuracy = float(correct_sum) / n
    return layer_pred, accuracy

def plot(image, prediction):
    fig, axes = plt.subplots(1, 1)
    axes.imshow(image.reshape(28,28), cmap = "binary")
    label = "Prediction: {}".format(prediction)
    axes.set_xlabel(label)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.show()
