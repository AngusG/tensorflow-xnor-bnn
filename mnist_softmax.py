# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
from gemm_op import xnor_gemm

FLAGS = None
N_HIDDEN = 512


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    with tf.name_scope('fc1_fp') as scope:
        W_1 = tf.Variable(tf.truncated_normal([784, N_HIDDEN]))
        b_1 = tf.Variable(tf.zeros([N_HIDDEN]))
        fc_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)

    with tf.name_scope('fc2_xnor') as scope:
        W_1 = tf.Variable(tf.truncated_normal([N_HIDDEN, N_HIDDEN]))
        b_1 = tf.Variable(tf.zeros([N_HIDDEN]))
        fc_1 = tf.nn.relu(gemm_module.gemm(x, W_1) + b_1)

    with tf.name_scope('fc3_fp') as scope:
        W_2 = tf.Variable(tf.truncated_normal([N_HIDDEN, 10]))
        b_2 = tf.Variable(tf.zeros([10]))
        y = tf.matmul(fc_1, W_2) + b_2

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train
    for step in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        __, loss = sess.run([train_step, cross_entropy],
                            feed_dict={x: batch_xs, y_: batch_ys})

        if step % 100 == 0:
            # Test trained model
            test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                     y_: mnist.test.labels})
            print("step %d, loss = %.4f, test accuracy %.4f" %
                  (step, loss, test_acc))

    # Test trained model
    print("Final test accuracy %.4f" % (sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                                      y_: mnist.test.labels})))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/scratch/gallowaa/mnist',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
