#!/usr/bin/python3
"""
author: qtz
time: 17-6-26
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# step 1: define algorithm formula, that is Neural network propagation calculation
# create default Session
sess = tf.InteractiveSession()
# create Variable
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# define softmax regression algorithm
y = tf.nn.softmax(tf.matmul(x, W) + b)

# step 2: define loss, choose Optimizer to optimize the loss
# define loss function
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# use Optimizer algorithm to train the model
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# step 3: iterative(迭代的,重复的) training
# initialize the global parameter
tf.global_variables_initializer().run()

# train the model
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs, y_:batch_ys})

# step 4: predict accuracy in test data or validation set
# validate the model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
print(correct_prediction)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy)
# compute accuracy
print(accuracy.eval({x:mnist.test.images, y_:mnist.test.labels}))