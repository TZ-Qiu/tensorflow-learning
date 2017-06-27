#!/usr/bin/python3
"""
author: qtz
time: 17-6-26
"""

import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def xavier_init(fan_in, fan_out, constant = 1):
    """
    xavier initialization, make weight size proper
    :param fan_in: number of input node
    :param fan_out: number of output node
    :param constant:
    :return:
    """
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


class AdditivdeGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function = tf.nn.softplus,
                 optimizer = tf.train.AdamOptimizer(), scale = 0.1):
        """
        Denoising Auto Encoder class
        :param n_input: number of input node
        :param n_hidden: number of hidden node
        :param transfer_function: activation function of hidden layer, default is softplus
        :param optimizer: optimizer(优化器) , default is Adam
        :param scale: Gauss noise coefficient(高斯噪声系数), default is 0.1
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        # define network structure
        self.x = tf.placeholder(tf.float32, [None, self.n_input])           # input data
        self.hidden = self.transfer(tf.add(tf.matmul(
                            self.x + scale * tf.random_normal((n_input,)),
                            self.weights['w1']), self.weights['b1']))       # hidden layer, used to extract the high order feature from raw data
        self.reconstruction = tf.add(tf.matmul(self.hidden,
                                               self.weights['w2']), self.weights['b2'])     # output layer(reconstructioin layer), restore high order feature into raw data
        # define AutoEncoder loss function
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(
            self.reconstruction, self.x), 2.0))
        # define Optimizer: optimize loss function
        self.optimizer = optimizer.minimize(self.cost)

        # initialize the global parameter
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        """
        parameter initializtion function
        :return: type: dict
        """
        all_weight = dict()
        all_weight['w1'] = tf.Variable(xavier_init(self.n_input,
                                                   self.n_hidden))      # input layer
        all_weight['b1'] = tf.Variable(tf.zeros([self.n_hidden],
                                                dtype=tf.float32))
        all_weight['w2'] = tf.Variable(tf.zeros([self.n_hidden,
                                self.n_input], dtype=tf.float32))       # output layer: without using activate function, so initialize all zero
        all_weight['b2'] = tf.Variable(tf.zeros([self.n_input],
                                                dtype=tf.float32))
        return all_weight

    def partial_fit(self, X):
        """
        train batch data, and compute loss cost
        :param X: input data
        :return: cost
        """
        cost, opt = self.sess.run((self.cost, self.optimizer),
                        feed_dict = {self.x: X, self.scale: self.training_scale})
        return cost

    def calc_total_cost(self, X):
        """
        only calculate cost, used to test data
        :param X: input data
        :return:
        """
        return self.sess.run(self.cost, feed_dict={self.x: X,
                                self.scale: self.training_scale})

    def transform(self, X):
        """
        activate functioin: extract the high order feature(从原始数据中提取高阶特征)
        :param X: raw data
        :return: return output value in hidden layer of AutoEncoder
        """
        return self.sess.run(self.n_hidden, feed_dict={self.x: X,
                             self.scale: self.training_scale})

    def generate(self, hidden=None):
        """
        restore(复原) high order feature into raw data(将提取到的高阶特征复原为原始数据)
        :param hidden:
        :return:
        """
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction,
                             feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        """
        run the whole restore process(contains two parts: transform + generate)
        :param X: raw data
        :return: the restored data
        """
        return self.sess.run(self.reconstruction, feed_dict={self.x: X,
                                self.scale: self.training_scale})

    def getWeithts(self):
        """
        get hidden layer weights
        :return:
        """
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        """
        get hidden layer biases
        :return:
        """
        return self.sess.run(self.weights['b1'])