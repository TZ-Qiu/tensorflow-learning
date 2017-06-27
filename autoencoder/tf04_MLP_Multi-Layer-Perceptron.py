#!/usr/bin/python3
"""
author: qtz
time: 17-6-26
"""

# FCN: Fully Connected Network

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# step 1: define algorithm formula, that is Neural network propagation calculation
# create default Session
sess = tf.InteractiveSession()

# initialize hidden layer and output layer parameter
in_units = 784          # input node number
h1_units = 300          # output node number of hidden layer
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))     # hidden layer weight
b1 = tf.Variable(tf.zeros([h1_units]))                                      # hidden layer biase
W2 = tf.Variable(tf.zeros([h1_units, 10]))                                  # output layer weight
b2 = tf.Variable(tf.zeros([10]))                                            # output layer biase

# input data placeholder
x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)              # for train: < 1(keep random, prevent over fitting);  for test: =1

# define model structure
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2) # output layer

# step 2: define loss, choose Optimizer to optimize the loss
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                                              reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

# step 3: iterative(迭代的,重复的) training
tf.global_variables_initializer().run()
for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})

# step 4: predict accuracy in test data or validation set
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels,
                     keep_prob: 1.0}))