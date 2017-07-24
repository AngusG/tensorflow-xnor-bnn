from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path
import argparse

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python import debug as tf_debug

import tensorflow as tf
from binary_net import BinaryNet

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
        '--binary', help="should weights and activations be constrained to -1, +1", action="store_true")
    parser.add_argument(
        '--fast', help="if binary flag is passed, determines if xnor_gemm cuda kernel is used to accelerate training, otherwise no effect", action="store_true")
    parser.add_argument(
        '--debug', help="run with tfdbg", action="store_true")
    args = parser.parse_args()

    # handle command line args
    if args.binary:
        binary = True
        sub_1 = '/bin/'
        if args.fast:
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

    #outpath = os.path.join(args.train_dir + sub_1 + sub_2, args.sub)
    outpath = os.path.join(args.train_dir + sub_1 + sub_2)

    # import data
    if binary:
        mnist = input_data.read_data_sets(args.data_dir, dtype=tf.uint8, one_hot=True)
    else:
        mnist = input_data.read_data_sets(args.data_dir, dtype=tf.float32, one_hot=True)        

    # create the model
    x = tf.placeholder(tf.float32, [None, 784])
    bnn = BinaryNet(binary, fast, args.n_hidden, x)
    y = bnn.output
    y_ = tf.placeholder(tf.float32, [None, 10])

    # define loss and optimizer
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    print("tf.InteractiveSession()")
    tf.global_variables_initializer().run()
    print("tf.global_variables_initializer().run()")

    if args.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    # setup summary writer
    summary_writer = tf.summary.FileWriter(outpath, sess.graph)
    training_summary = tf.summary.scalar("train loss", cross_entropy)
    test_summary = tf.summary.scalar("test acc.", accuracy)

    # Train
    for step in range(1000):

        batch_xs, batch_ys = mnist.train.next_batch(512)
        __, loss, train_summ = sess.run([train_step, cross_entropy, training_summary],
                                        feed_dict={x: batch_xs, y_: batch_ys})

        if step % 100 == 0:

            # Test trained model
            if binary:
                test_batch_xs, test_batch_ys = mnist.test.next_batch(512)
                test_acc, test_summ = sess.run([accuracy, test_summary], feed_dict={
                    x: test_batch_xs, y_: test_batch_ys})
            else:
                test_acc, test_summ = sess.run([accuracy, test_summary], feed_dict={x: mnist.test.images,
                                                                         y_: mnist.test.labels})
            print("step %d, loss = %.4f, test accuracy %.4f" %
                  (step, loss, test_acc))

            summary_writer.add_summary(train_summ, step)
            summary_writer.add_summary(test_summ, step)
            summary_writer.flush()

    # Test trained model
    print("Final test accuracy %.4f" % (sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                                      y_: mnist.test.labels})))
