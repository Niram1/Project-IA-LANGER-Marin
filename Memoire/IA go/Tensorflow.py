import gzip
import numpy as np
import six.moves.cPickle as pickle


def encode_label(j):
    e= np.zeros((10, 1))
    e[j] = 1.0
    return e

def shape_data(data):
    features = [np.reshape(x, (784, 1)) for x in data[0]]
    labels = [encode_label(y) for y in data[1]]
    return zip(features, labels)

def load_data():
    with gzip.open('C:\\Users\\Marin\\Documents\\Université\\Memoire\\IA go\\mnist.pkl.gz', 'rb') as f:
        train_data, validation_data, test_data = pickle.load(f)
    return shape_data(train_data), shape_data(test_data)

import numpy as np
from dlgo.nn.load_mnist import load_data
from dlgo.nn.layers import sigmoid_double

def average_digit(data, digit):
    filtered_data = [x[0] for x in data if np.argmax(x[1]) == digit]
    filtered_array = np.asarray(filtered_data)
    return np.average(filtered_array, axis=0)

train, test = load_data()
avg_eight = average_digit(train, 8)

from matplotlib import pyplot as plt
img = (np.reshape(avg_eight, (28, 28)))
plt.imshow(img)
plt.show()