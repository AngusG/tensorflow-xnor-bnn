from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import os.path
import argparse

from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python import debug as tf_debug

import numpy as np
import tensorflow as tf

from binary_net import BinaryNet
from utils import create_dir_if_not_exists

BN_TRAIN_PHASE = 1
BN_TEST_PHASE = 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    '''
    parser.add_argument(
        'sub', help='sub-directory under --train_dir for logging events and checkpointing.   \
        Would usually give a unique name (e.g initial learning rate used) so that tensorboard \
        results are more easily interpreted')
    '''
    parser.add_argument('--data_dir', type=str, default='/scratch/gallowaa/mnist',
                        help='directory for storing input data')
    parser.add_argument('--train_dir', type=str, default='/scratch/gallowaa/logs/tf-bnn',
                        help='root path for logging events and checkpointing')
    parser.add_argument(
        '--n_hidden', help='number of hidden units', type=int, default=512)
    parser.add_argument(
        '--reg', help='l1 regularization penalty', type=float, default=0.0001)
    parser.add_argument(
        '--lr', help='learning rate', type=float, default=0.00001)
    parser.add_argument(
        '--batch_size', help='examples per mini-batch', type=int, default=100)
    parser.add_argument(
        '--max_steps', help='maximum training steps', type=int, default=1000)
    parser.add_argument(
        '--eval_every_n', help='validate model every n steps', type=int, default=100)
    parser.add_argument(
        '--binary', help="should weights and activations be constrained to -1, +1", action="store_true")
    parser.add_argument(
        '--fast', help="if binary flag is passed, determines if xnor_gemm cuda kernel is used to accelerate training, otherwise no effect", action="store_true")
    parser.add_argument(
        '--batch_norm', help="batch normalize activations", action="store_true")
    parser.add_argument(
        '--summ', help="log summaries of weights, activations, gradients for viewing in TensorBoard", action="store_true")
    parser.add_argument(
        '--debug', help="run with tfdbg", action="store_true")
    args = parser.parse_args()

    # handle command line args
    if args.binary:
        print("Using 1-bit weights and activations")
        binary = True
        sub_1 = '/bin/'
        if args.fast:
            print("Using fast xnor_gemm kernel")
            fast = True
            sub_2 = 'xnor/'
        else:
            sub_2 = 'matmul/'
            fast = False
    else:
        sub_1 = '/fp/'
        sub_2 = ''
        binary = False
        fast = False

    log_path = args.train_dir + sub_1 + \
        sub_2 + 'hid_' + str(args.n_hidden) + '/'
    if args.batch_norm:
        print("Using batch normalization")
        batch_norm = True
        alpha = 0.1
        epsilon = 1e-4
        log_path += 'batch_norm/'
    else:
        batch_norm = False

    if args.summ:
        log_path += 'bs_' + str(args.batch_size)
        log_path = os.path.join(log_path, str(args.reg))
        log_path = create_dir_if_not_exists(log_path)

    # import data
    mnist = input_data.read_data_sets(
        args.data_dir, dtype=tf.float32, one_hot=True)
    dtype = tf.float32

    global_step = variable_scope.get_variable(  # needs to be defined for tf.contrib.layers.optimize_loss()
        "global_step", [],
        trainable=False,
        dtype=dtypes.int64,
        initializer=init_ops.constant_initializer(0, dtype=dtypes.int64))

    x = tf.placeholder(dtype, [None, 784])
    phase = tf.placeholder(tf.bool, name='phase')

    # create the model
    bnn = BinaryNet(binary, fast, args.n_hidden, x, batch_norm, phase)
    y = bnn.output
    y_ = tf.placeholder(tf.float32, [None, 10])

    # define loss and optimizer
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    l1_regularizer = tf.contrib.layers.l1_regularizer(scale=args.reg)
    weights = tf.trainable_variables()  # all vars of your graph
    reg = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
    total_loss = cross_entropy + reg

    # for batch-normalization
    if batch_norm:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # ensures that we execute the update_ops before performing the train_op
            #train_op = tf.train.AdamOptimizer(0.01).minimize(total_loss)
            train_op = tf.contrib.layers.optimize_loss(
                total_loss, global_step, learning_rate=0.001, optimizer='Adam',
                summaries=["gradients"])
    else:
        #train_op = tf.train.AdamOptimizer(0.01).minimize(total_loss)
        train_op = tf.contrib.layers.optimize_loss(
            total_loss, global_step, learning_rate=0.001, optimizer='Adam',
            summaries=["gradients"])

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if args.debug:
        print("Using debug mode")
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    # setup summary writer
    if args.summ:
        summary_writer = tf.summary.FileWriter(log_path, sess.graph)
        training_summary = tf.summary.scalar("train loss", total_loss)
        test_summary = tf.summary.scalar("test acc.", accuracy)
        merge_op = tf.summary.merge_all()

    # Train
    timing_arr = np.zeros(args.max_steps)
    for step in range(args.max_steps):

        batch_xs, batch_ys = mnist.train.next_batch(args.batch_size)

        start_time = time.time()
        __, loss = sess.run([train_op, total_loss], feed_dict={
            x: batch_xs, y_: batch_ys, phase: BN_TRAIN_PHASE})
        timing_arr[step] = time.time() - start_time

        if step % args.eval_every_n == 0:

            if fast:
                test_batch_xs, test_batch_ys = mnist.test.next_batch(
                    args.batch_size)
                if args.summ:
                    test_acc, merged_summ = sess.run([accuracy, merge_op], feed_dict={
                        x: test_batch_xs, y_: test_batch_ys, phase: BN_TEST_PHASE})
                else:
                    test_acc = sess.run(accuracy, feed_dict={
                        x: test_batch_xs, y_: test_batch_ys, phase: BN_TEST_PHASE})
            else:
                if args.summ:
                    test_acc, merged_summ = sess.run([accuracy, merge_op], feed_dict={
                        x: mnist.test.images, y_: mnist.test.labels, phase: BN_TEST_PHASE})
                else:
                    test_acc = sess.run(accuracy, feed_dict={
                        x: mnist.test.images, y_: mnist.test.labels, phase: BN_TEST_PHASE})
            print("step %d, loss = %.4f, test accuracy %.4f (%.1f ex/s)" %
                  (step, loss, test_acc, float(args.batch_size / timing_arr[step])))

            if args.summ:
                summary_writer.add_summary(merged_summ, step)
                summary_writer.flush()

    # Test trained model
    if not fast:
        print("Final test accuracy %.4f" % (sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                                          y_: mnist.test.labels,
                                                                          phase: BN_TEST_PHASE})))
    print("Avg ex/s = %.1f" % float(args.batch_size / np.mean(timing_arr)))
    print("Med ex/s = %.1f" % float(args.batch_size / np.median(timing_arr)))
