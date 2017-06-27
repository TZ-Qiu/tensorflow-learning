#!/usr/bin/python3
"""
author: qtz
time: 17-6-26
"""

import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from autoencoder import tf03_denoising_autoencoder

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def standard_scale(X_train, X_test):
    """
    Normalization(标准化处理),使数据成为均值为0,标准差为1的分布
    :param X_train:
    :param X_test:
    :return:
    """
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

def get_random_block_from_data(data, batch_size):
    """
    get random block data(不放回抽样，提高数据利用率)
    :param data:
    :param batch_size:
    :return:
    """
    start_index = np.random.randint(0, len(data)-batch_size)
    return data[start_index:(start_index + batch_size)]

# normalize train data and test data
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

# define frequently-used parameter
# total samples number
n_samples = int(mnist.train.num_examples)
# 训练轮数
training_epochs = 20
batch_size = 128
# 每隔一轮显示一次损失cost
display_step = 1

# instantiate autoencoder
autoencoder = tf03_denoising_autoencoder.AdditivdeGaussianNoiseAutoencoder(n_input=784,
                                            n_hidden=200,
                                            transfer_function = tf.nn.softplus,
                                            optimizer = tf.train.AdamOptimizer(learning_rate=0.001),
                                            scale=0.01)
# train
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)

        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost / n_samples * batch_size

    # after iterate each batch, show current iterate number and avg_cost
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=",
                        "{:.9f}".format(avg_cost))

    # test performance
    print("Total cost:" + str(autoencoder.calc_total_cost(X_test)))