#!/usr/bin/python3
"""
author: qtz
time: 17-6-28
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

max_steps = 1000
learning_rate = 0.001
dropout = 0.9
data_dir = '/tmp/tensorflow/mnist/input_data'
log_dir = '/tmp/tensorflow/mnist/logs/mnist_with_summaries'

# download mnist data
mnist = input_data.read_data_sets(data_dir, one_hot=True)
sess = tf.InteractiveSession()

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None,784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None,10], name='y-input')

# transform input data of one dimension into 28*28 image, and storage to another tensor
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1,28,28,1])
    tf.summary.image('input', image_shaped_input, 10)

def weight_variable(shape):
    """
    define neural network parameter initialization method
    :param shape:
    :return:
    """
    inital = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(inital)

def bias_variable(shape):
    inital = tf.constant(0.1, shape=shape)
    return tf.Variable(inital)

def variable_summaries(var):
    """
    define data aggregate function(汇总函数) of var
    :param var:
    :return:
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        tf.summary.scalar('stddev', stddev)     # record and summary(汇总)
        tf.summary.scalar('max', tf.reduce_mean(var))
        tf.summary.scalar('min', tf.reduce_mean(var))
        tf.summary.histogram('histogram', var)  # record histogram(直方图) data of var

def nn_lay(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """
    design a multi-layer MLP neural network to train the data, and summary model parameter in eath layer
    :param input_tensor: input data
    :param input_dim: input data dimension
    :param output_dim: output data dimension
    :param layer_name:
    :param act: activation func, default is ReLU
    :return:
    """
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)    # before activation
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)            # after activation
        return activations

# define hidden layer neural network
hidden1 = nn_lay(x, 784, 500, 'layer1')     # input dimension: 784/28*28;  output dimension: 500 numbers of hidden node;

# create dropout layer
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

# define output layer neural network
y = nn_lay(dropped, 500, 10, 'layer2', act=tf.identity)     # input dimension: 500;  output dimension: 10/classification number;

# softmax layer  compute loss value
with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

# use optimizer to optimize loss value
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# count correct samples and compute accuracy
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)


merged = tf.summary.merge_all()         # get all summary operation
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)    # storage log of train, add sess.graph to visualize(可视化)
test_writer = tf.summary.FileWriter(log_dir + '/test')      # storage log of test
tf.global_variables_initializer().run()

def feed_dict(train):
    """
    define deed_dict loss func
    :param train: True of False
    :return:
    """
    if train:
        xs, ys = mnist.train.next_batch(100)
        k = dropout
    else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
    return {x: xs, y_: ys, keep_prob: k}


# train, test, record log
saver = tf.train.Saver()        # create model save container
for i in range(max_steps):
    if i % 10 == 0:
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))     # run merged and accuracy operation per 10 step
        test_writer.add_summary(summary, i)
        print('Accuracy at step %s: %s' % (i, acc))
    else:
        if i % 100 == 99:
            run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)     # run options(运行选项) per 100 step
            run_metadata = tf.RunMetadata()     # run metadata(运行的元信息)  convenient to record run time and memory usage information in training
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True),
                                  options=run_options, run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)     # add run_metadata to train_writer
            train_writer.add_summary(summary, i)            # add summary to train_writer
            saver.save(sess, log_dir+"/model.ckpt", i)
            print('Adding run metadata for', i)
        else:
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))  # only run merged and train_step operation
            train_writer.add_summary(summary, i)

train_writer.close()        # after training
test_writer.close()