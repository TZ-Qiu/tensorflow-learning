#!/usr/bin/python3
"""
author: qtz
time: 17-6-27
"""

from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
num_batches = 100

def print_activations(t):
    """
    show the structure of the network in each layer
    to be specific, display output tensor name and size of each conv layer or pool layer
    :param t: tensor
    :return: None
    """
    print(t.op.name, ' ', t.get_shape().as_list())


def inference(images):
    """
    design AlexNet network structure
    :param images:
    :return: pool5 layer and parameters
    """
    parameters = []     # storage model parameter required training in AlexNet

    # conv1 layer
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11,11,3,64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')  # initialize conv kernel parameter: conv kernel size: 11*11; color channel: 3;  conv number:64
        conv = tf.nn.conv2d(images, kernel, [1,4,4,1], padding='SAME')          # strides: 4*4
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)        # print tensor conv1 structure
        parameters += [kernel, biases]  # storage model parameter required training in AlexNet

    # lrn1 layer and pool1 layer
    lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn1')
    pool1 = tf.nn.max_pool(lrn1, ksize=[1,3,3,1], strides=[1,2,2,1],
                                                padding='VALID', name='pool1')  # pool size: 3*3, 3*3--->1*1;  strides:2*2
    print_activations(pool1)

    # conv2 layer
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5,5,64,192], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')  # initialize conv kernel parameter: conv kernel size: 5*5; color channel: 64;  conv number:192
        conv = tf.nn.conv2d(pool1, kernel, [1,1,1,1], padding='SAME')           # strides: 1*1
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]  # storage model parameter required training in AlexNet
    print_activations(conv2)  # print tensor conv2 structure

    # lrn2 layer and pool2 layer
    lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn2')
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool2')  # pool size: 3*3, 3*3--->1*1;  strides:2*2
    print_activations(pool2)

    # conv3 layer
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')  # initialize conv kernel parameter: conv kernel size: 3*3; color channel: 192;  conv number:384
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')  # strides: 1*1
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]  # storage model parameter required training in AlexNet
    print_activations(conv3)  # print tensor conv3 structure

    # conv4 layer
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')  # initialize conv kernel parameter: conv kernel size: 3*3; color channel: 384;  conv number:256
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')  # strides: 1*1
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]  # storage model parameter required training in AlexNet
    print_activations(conv4)  # print tensor conv4 structure

    # conv5 layer
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')  # initialize conv kernel parameter: conv kernel size: 3*3; color channel: 384;  conv number:256
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')  # strides: 1*1
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]  # storage model parameter required training in AlexNet
    print_activations(conv5)  # print tensor conv5 structure

    # pool5 layer
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool5')  # pool size: 3*3, 3*3--->1*1;  strides:2*2
    print_activations(pool5)

    return pool5, parameters


def time_tensorflow_run(session, target, info_string):
    """
    evaluate computation time of each round
    :param session: Session
    :param target: op
    :param info_string: test name
    :return: None
    """
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0

    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:              # to begin, warm up num_steps_burn_in iteration
            if not i % 10:
                print('%s: step %d, duration = %.3f' % (datetime.now(),
                                                        i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration

    # after iteration, calculate the average time consumption and standard deviation(标准差)
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
          (datetime.now(), info_string, num_batches, mn, sd))


def run_benchmark():
    """
    main func
    without using ImageNet data set to train, only use random images data to test the consumption time of Forward and backward propagation
    :return: None
    """
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3],
                                              dtype=tf.float32,stddev=1e-1))
        pool5, parameters = inference(images)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

    # count forward computing time
    time_tensorflow_run(sess, pool5, "Forward")

    # count backward computing time
    objective = tf.nn.l2_loss(pool5)            # set optimized object
    grad = tf.gradients(objective, parameters)  # get the gradient for all model parameters
    time_tensorflow_run(sess, grad, "Forward-backward")

run_benchmark()