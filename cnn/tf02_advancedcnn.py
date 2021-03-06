#!/usr/bin/python3
"""
author: qtz
time: 17-6-27
"""

from cnn import cifar10,cifar10_input
import tensorflow as tf
import numpy as np
import time

max_steps = 3000
batch_size = 128
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'

# step 1: define algorithm formula, that is Neural network propagation calculation
# Preparation work
def variable_with_weight_loss(shape, stddev, w1):
    """
    initialize weight func
    :param shape:
    :param stddev:
    :param w1: weight loss value
    :return:
    """
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

# use cifar10 class to download, unzip and extract the data set to the default location
cifar10.maybe_download_and_extract()

# generate train data required to use   Data Augmentation(数据增强)
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)

# generate test data
images_test, labels_test = cifar10_input.inputs(eval_data=True,
                                                data_dir=data_dir, batch_size=batch_size)

# create input data placeholder, contains feature and label
image_holder = tf.placeholder(tf.float32, [batch_size,24,24,3])       # batch_size: specified, not None; image size: 24 x 24(裁剪后的大小); 3: color channel
label_hoder = tf.placeholder(tf.int32, [batch_size])

# define model structure
# first conv layer
weight1 = variable_with_weight_loss(shape=[5,5,3,64], stddev=5e-2, w1=0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, [1,1,1,1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

# second conv layer
weight2 = variable_with_weight_loss(shape=[5,5,64,64], stddev=5e-2, w1=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1,1,1,1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

# first fcn layer
reshape = tf.reshape(pool2, [batch_size, -1])       # flatten, reshape conv layer output vector from 2D to 1D
dim = reshape.get_shape()[1].value                  # after flattening, get its length
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)    # hidden layer node number: 384
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

# second fcn layer
weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)    # hidden layer node number: 192
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

# last layer        note: not use softmax to output classification results
weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1/192.0, w1=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.nn.relu(tf.matmul(local4, weight5) + bias5)

def loss(logits, labels):
    """
    compute total loss
    :param logits:
    :param labels:
    :return:  total loss, contains cross entropy loss, and weight L2 loss of the last two layer
    """
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')

# step 2: define loss, choose Optimizer to optimize the loss
# get loss value
loss = loss(logits, label_hoder)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

# compute top k accuracy of the output results, default is top 1
top_k_op = tf.nn.in_top_k(logits, label_hoder, 1)

# create default Session and initialize the global parameter
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# start thread queue of the image data augmentation (use 16 threads to accelerate)    important step
tf.train.start_queue_runners()

# step 3: iterative(迭代的,重复的) training
# train the model
for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])   # get batch size train data
    _, loss_value = sess.run([train_op, loss],
                             feed_dict={image_holder: image_batch, label_hoder: label_batch})
    duration = time.time() - start_time

    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)

        format_str = ('step %d,loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))     # loss_value: current loss value; examples_per_sec: the number of training samples per second; sec_per_batch: the time of training a batch data

# step 4: predict accuracy in test data or validation set
num_examples = 10000        # test set sample number
import math
num_iter = int(math.ceil(num_examples / batch_size))    # batch number
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])     # get images_test batch and labels_test batch
    predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch,
                                                  label_hoder: label_batch})    # compute sample numbers of the correct prediction in the top 1 batch of the model
    true_count += np.sum(predictions)
    step += 1

# compute accuracy and print results
precision = true_count / total_sample_count
print('precision @ 1 = %.3f' % precision)