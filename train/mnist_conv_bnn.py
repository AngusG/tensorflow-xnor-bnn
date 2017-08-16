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

from models.binary_conv_net import BinaryConvNet
from utils import create_dir_if_not_exists

BN_TRAIN_PHASE = True
BN_TEST_PHASE = False

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='directory for storing input data')
    parser.add_argument(
        '--log_dir', help='root path for logging events and checkpointing')
    parser.add_argument(
        '--n_hidden', help='number of hidden units', type=int, default=512)
    parser.add_argument(
        '--keep_prob', help='dropout keep_prob', type=float, default=0.8)
    parser.add_argument(
        '--lr', help='learning rate', type=float, default=1e-5)
    parser.add_argument(
        '--batch_size', help='examples per mini-batch', type=int, default=128)
    parser.add_argument(
        '--max_steps', help='maximum training steps', type=int, default=1000)
    parser.add_argument(
        '--gpu', help='physical id of GPUs to use')
    parser.add_argument(
        '--eval_every_n', help='validate model every n steps', type=int, default=100)
    parser.add_argument(
        '--binary', help="should weights and activations be constrained to -1, +1", action="store_true")
    parser.add_argument(
        '--xnor', help="if binary flag is passed, determines if xnor_gemm cuda kernel is used to accelerate training, otherwise no effect", action="store_true")
    parser.add_argument(
        '--batch_norm', help="batch normalize activations", action="store_true")
    parser.add_argument(
        '--debug', help="run with tfdbg", action="store_true")
    parser.add_argument(
        '--restore', help='where to load model checkpoints from')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # handle command line args
    if args.binary:
        print("Using 1-bit weights and activations")
        binary = True
        sub_1 = '/bin/'
        if args.xnor:
            print("Using xnor xnor_gemm kernel")
            xnor = True
            sub_2 = 'xnor/'
        else:
            sub_2 = 'matmul/'
            xnor = False
    else:
        sub_1 = '/fp/'
        sub_2 = ''
        binary = False
        xnor = False

    if args.log_dir:
        log_path = args.log_dir + sub_1 + sub_2 + \
            'hid_' + str(args.n_hidden) + '/'

    if args.batch_norm:
        print("Using batch normalization")
        batch_norm = True
        alpha = 0.1
        epsilon = 1e-4
        if args.log_dir:
            log_path += 'batch_norm/'
    else:
        batch_norm = False

    if args.log_dir:
        log_path += 'bs_' + str(args.batch_size) + '/keep_' + \
            str(args.keep_prob) + '/lr_' + str(args.lr)
        log_path = create_dir_if_not_exists(log_path)

    # import data
    mnist = input_data.read_data_sets(
        args.data_dir, dtype=tf.float32, one_hot=True)
    dtype = tf.float32

    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        x = tf.placeholder(dtype, [None, 28, 28, 1])
        phase = tf.placeholder(tf.bool, name='phase')
        keep_prob = tf.placeholder_with_default(1.0, shape=(None))

        # create the model
        bnn = BinaryConvNet(binary, xnor, args.n_hidden,
                            keep_prob, x, batch_norm, phase)
        y = bnn.output
        y_ = tf.placeholder(tf.float32, [None, 10])

        # define loss and optimizer
        total_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

        # for batch-normalization
        if batch_norm:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # ensures that we execute the update_ops before performing the
                # train_op
                train_op = tf.contrib.layers.optimize_loss(
                    total_loss, global_step, learning_rate=args.lr, optimizer='Adam',
                    summaries=["gradients"])
        else:
            train_op = tf.contrib.layers.optimize_loss(
                total_loss, global_step, learning_rate=args.lr, optimizer='Adam',
                summaries=["gradients"])
        #train_op = tf.train.AdamOptimizer(args.lr).minimize(total_loss, global_step=global_step)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver(tf.global_variables())

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        if args.debug:
            print("Using debug mode")
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        # setup summary writer
        if args.log_dir:
            summary_writer = tf.summary.FileWriter(log_path, sess.graph)
            training_summary = tf.summary.scalar("train loss", total_loss)
            test_summary = tf.summary.scalar("test acc.", accuracy)
            merge_op = tf.summary.merge_all()

        if args.restore:
            saver.restore(sess, tf.train.latest_checkpoint(args.restore))
            # os.path.join(log_path, args.restore)))
            init_step = sess.run(global_step)
            print('Restoring network previously trained to step %d' % init_step)
        else:
            init_step = 0

        # Train
        timing_arr = np.zeros(args.max_steps)
        step = init_step
        while step < init_step + args.max_steps:

            batch_xs, batch_ys = mnist.train.next_batch(args.batch_size)
            batch_xs = batch_xs.reshape(args.batch_size, 28, 28, 1)
            batch_ys = batch_ys.reshape(args.batch_size, 10)

            start_time = time.time()
            __, loss = sess.run([train_op, total_loss], feed_dict={
                x: batch_xs, y_: batch_ys, keep_prob: args.keep_prob, phase: BN_TRAIN_PHASE})
            timing_arr[step - init_step] = time.time() - start_time

            if step > args.eval_every_n and step % args.eval_every_n == 0:

                if xnor:
                    test_batch_xs, test_batch_ys = mnist.test.next_batch(
                        args.batch_size)
                    test_batch_xs = test_batch_xs.reshape(
                        args.batch_size, 28, 28, 1)
                    test_batch_ys = test_batch_ys.reshape(args.batch_size, 10)
                    if args.log_dir:
                        test_acc, merged_summ = sess.run([accuracy, merge_op], feed_dict={
                            x: test_batch_xs, y_: test_batch_ys, keep_prob: 1.0, phase: BN_TEST_PHASE})
                    else:
                        test_acc = sess.run(accuracy, feed_dict={
                            x: test_batch_xs, y_: test_batch_ys, phase: BN_TEST_PHASE})
                else:
                    if args.log_dir:
                        test_acc, merged_summ = sess.run([accuracy, merge_op], feed_dict={
                            x: mnist.test.images.reshape(10000, 28, 28, 1), y_: mnist.test.labels.reshape(10000, 10), keep_prob: 1.0, phase: BN_TEST_PHASE})
                    else:
                        test_acc = sess.run(accuracy, feed_dict={
                            x: mnist.test.images.reshape(10000, 28, 28, 1), y_: mnist.test.labels.reshape(10000, 10), keep_prob: 1.0, phase: BN_TEST_PHASE})
                print("step %d, loss = %.4f, test accuracy %.4f (%.1f ex/s)" %
                      (step, loss, test_acc, float(args.batch_size / timing_arr[step - init_step])))

                if args.log_dir:
                    summary_writer.add_summary(merged_summ, step)
                    summary_writer.flush()
            step += 1

        # Test trained model
        if not xnor:
            print("Final test accuracy %.4f" % (sess.run(accuracy, feed_dict={x: mnist.test.images.reshape(10000, 28, 28, 1),
                                                                              y_: mnist.test.labels.reshape(10000, 10),
                                                                              keep_prob: 1.0,
                                                                              phase: BN_TEST_PHASE})))
        print("Avg ex/s = %.1f" % float(args.batch_size / np.mean(timing_arr)))
        print("Med ex/s = %.1f" %
              float(args.batch_size / np.median(timing_arr)))

        if args.log_dir:
            # save model checkpoint
            checkpoint_path = os.path.join(log_path, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
            sess.close()
