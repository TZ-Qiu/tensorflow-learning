#!/usr/bin/python3
"""
author: qtz
time: 17-6-26
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# plot one example
print(mnist.train.images.shape, mnist.train.labels.shape)     # (55000, 28 * 28)  (55000, 10)
print(mnist.test.images.shape, mnist.test.labels.shape)       # (10000, 784) (10000, 10)
print(mnist.validation.images.shape, mnist.validation.labels.shape) # (5000, 784) (5000, 10)
print(mnist.train.images)