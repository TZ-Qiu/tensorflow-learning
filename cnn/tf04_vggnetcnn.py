#!/usr/bin/python3
"""
author: qtz
time: 17-6-28
"""

from datetime import datetime
import math
import time
import tensorflow as tf

def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    """
    create convolution layer, and storage this layer model parameter into parameter list
    :param input_op: input tensor
    :param name: this layer name
    :param kh: conv kernel height
    :param kw: conv kernel width
    :param n_out: conv number, i.e. output channel number
    :param dh: strides height
    :param dw: strides width
    :param p: parameter list
    :return: activation
    """
    n_in = input_op.get_shape()[-1].value       # get input_op input channel number    such as: 224*224*3  the end of 3

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w", shape=[kh,kw,n_in,n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())     # note: using tf.get_variable to create kernel, not use tf.Variable; use xavier to initialize parameter
        conv = tf.nn.conv2d(input_op, kernel, (1,dh,dw,1), padding='SAME')          # strides: dh*dw
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')               # transform bias_init_val variable into trainable parameter
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        return activation


def fc_op(input_op, name, n_out, p):
    """
    define fcn layer creation func
    :param input_op: input tensor
    :param name: this layer name
    :param n_out: conv number, i.e. output channel number
    :param p: parameter list
    :return: activation
    """
    n_in = input_op.get_shape()[-1].value          # get input_op input channel number

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w",
                                 shape=[n_in,n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())    # use tf.get_variable to create fcn layer parameter    the shape is two dimension
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        p += [kernel, biases]
        return activation

def mpool_op(input_op, name, kh, kw, dh, dw):
    """
    define max pool layer creation func
    :param input_op:
    :param name:
    :param kh:
    :param kw:
    :param dh:
    :param dw:
    :return:
    """
    return tf.nn.max_pool(input_op,
                          ksize=[1,kh,kw,1],
                          strides=[1,dh,dw,1],
                          padding='SAME',
                          name=name)                # max pool size: kh*kw;  strides:dh*dw


def inference_op(input_op, keep_prob):
    """
    create VGGNet-16 network structure
    :param input_op: input tensor
    :param keep_prob: a placeholder to control dropout rate
    :return:
    """
    p = []

    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1,
                      p=p)  # conv1_1 kernel size: 3*3; conv number(ouput channel number):64; strides:1*1;    input_op size: 224*224*3; output size: 224*224*64
    conv1_2 = conv_op(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1,
                      p=p)  # input_op size: 224*224*64; output size: 224*224*64
    pool1 = mpool_op(conv1_2, name="pool1", kh=2, kw=2, dw=2, dh=2)  # max pool size: 2*2; output size: 112*112*64


    conv2_1 = conv_op(pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1,
                      p=p)  # conv2_1 kernel size: 3*3; conv number(ouput channel number):128; strides:1*1;    input_op size: 112*112*64; output size: 112*112*128
    conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1,
                      p=p)  # input_op size: 112*112*128; output size: 112*112*128
    pool2 = mpool_op(conv2_2, name="pool2", kh=2, kw=2, dw=2, dh=2)  # max pool size: 2*2; output size: 56*56*128


    conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1,
                      p=p)  # conv3_1 kernel size: 3*3; conv number(ouput channel number):256; strides:1*1;    input_op size: 56*56*128; output size: 56*56*256
    conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1,
                      p=p)  # input_op size: 56*56*256; output size: 56*56*256
    conv3_3 = conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1,
                      p=p)  # input_op size: 56*56*256; output size: 56*56*256
    pool3 = mpool_op(conv3_3, name="pool3", kh=2, kw=2, dw=2, dh=2)  # max pool size: 2*2; output size: 28*28*256


    conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1,
                      p=p)  # conv4_1 kernel size: 3*3; conv number(ouput channel number):512; strides:1*1;    input_op size: 28*28*256; output size: 28*28*512
    conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1,
                      p=p)  # input_op size: 28*28*512; output size: 28*28*512
    conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1,
                      p=p)  # input_op size: 28*28*512; output size: 28*28*512
    pool4 = mpool_op(conv4_3, name="pool4", kh=2, kw=2, dw=2, dh=2)  # max pool size: 2*2; output size: 14*14*512


    conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1,
                      p=p)  # conv5_1 kernel size: 3*3; conv number(ouput channel number):512; strides:1*1;    input_op size: 14*14*512; output size: 14*14*512
    conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1,
                      p=p)  # input_op size: 14*14*512; output size: 14*14*512
    conv5_3 = conv_op(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1,
                      p=p)  # input_op size: 14*14*512; output size: 14*14*512
    pool5 = mpool_op(conv5_3, name="pool5", kh=2, kw=2, dw=2, dh=2)  # max pool size: 2*2; strides: 2*2; output size: 7*7*512

    # transform each samples into one dimension vector(length: 7*7*512=25088)
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")   # -1: sample number unfixed(不固定)

    # fc layer: connect to 4096 numbers of hidden layer output node
    fc6 = fc_op(resh1, name="fc6", n_out=4096, p=p)
    # connect to dropout layer
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")

    # fc layer: connect to 4096 numbers of hidden layer output node
    fc7 = fc_op(fc6_drop, name="fc7", n_out=4096, p=p)
    # connect to dropout layer
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

    # fc layer: connect to 1000 numbers of hidden layer output node
    fc8 = fc_op(fc7_drop, name="fc8", n_out=1000, p=p)
    softmax = tf.nn.softmax(fc8)            # get classification probability(分类概率)
    predictions = tf.argmax(softmax, 1)     # get max classification(类别)
    return predictions, softmax, fc8, p


def time_tensorflow_run(session, target, feed,  info_string):
    """
    evaluate computation time of each round
    :param session: session
    :param target: op
    :param feed: convenient to control dropout layer retention rate(保留比率)
    :param info_string: test name
    :return:
    """
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0

    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target, feed_dict=feed)
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
                                              dtype=tf.float32,stddev=1e-1))    # get size of 224*224 random images
        keep_prob = tf.placeholder(tf.float32)
        predictions, softmax, fc8, p = inference_op(images, keep_prob)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

    # count forward computing time
    time_tensorflow_run(sess, predictions, {keep_prob: 1.0}, "Forward")

    # count backward computing time
    objective = tf.nn.l2_loss(fc8)            # set optimized object
    grad = tf.gradients(objective, p)  # get the gradient for all model parameters
    time_tensorflow_run(sess, grad, {keep_prob: 0.5}, "Forward-backward")


batch_size = 32
num_batches = 100
run_benchmark()