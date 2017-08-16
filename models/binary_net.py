import tensorflow as tf
from tf_gemm_op import xnor_gemm

BN_EPSILON = 1e-4


@tf.RegisterGradient("QuantizeGrad")
def quantize_grad(op, grad):
    return tf.clip_by_value(tf.identity(grad), -1, 1)


class BinaryNet:

    def __init__(self, binary, fast, n_hidden, keep_prob, x, batch_norm, phase):
        self.binary = binary
        self.fast = fast
        self.n_hidden = n_hidden
        self.keep_prob = keep_prob
        self.input = x
        self.G = tf.get_default_graph()
        self.dense_layers(batch_norm, phase)

    def init_layer(self, name, n_inputs, n_outputs):

        W = tf.get_variable(name, shape=[
                            n_inputs, n_outputs], initializer=tf.contrib.layers.xavier_initializer())
        #b = tf.Variable(tf.zeros([n_outputs]))
        return W

    def hard_sigmoid(self, x):
        return tf.clip_by_value((x + 1.) / 2, 0, 1)

    def binary_tanh_unit(self, x):
        return 2 * self.hard_sigmoid(x) - 1

    def quantize(self, x):
        with self.G.gradient_override_map({"Sign": "QuantizeGrad"}):
            return tf.sign(x)
            #E = tf.reduce_mean(tf.abs(x))
            # return tf.sign(x) * E

    def dense_layers(self, batch_norm, phase):

        if self.binary:

            with tf.name_scope('fc1_b') as scope:

                # don't quantize weights in first layer
                W_1 = self.init_layer('W_1', 784, self.n_hidden)
                self.w1_summ = tf.summary.histogram(name='W1_summ', values=W_1)

                fc1 = tf.nn.dropout(self.quantize(tf.matmul(self.input, W_1)), self.keep_prob)
                self.fc1_summ = tf.summary.histogram(
                    name='a1_summ', values=fc1)

            with tf.name_scope('fc2_b') as scope:

                W_2 = self.init_layer('W_2', self.n_hidden, self.n_hidden)
                Wb_2 = self.quantize(W_2)
                self.w2_summ = tf.summary.histogram(name='W2_summ', values=W_2)
                self.wb2_summ = tf.summary.histogram(
                    name='Wb2_summ', values=Wb_2)

                if self.fast:
                    fc2 = tf.nn.dropout(self.quantize(xnor_gemm(fc1, Wb_2)), self.keep_prob)
                else:
                    fc2 = tf.nn.dropout(self.quantize(tf.matmul(fc1, Wb_2)), self.keep_prob)

                self.fc2_summ = tf.summary.histogram(
                    name='a2_summ', values=fc2)

            with tf.name_scope('fc3_b') as scope:

                W_3 = self.init_layer('W_3', self.n_hidden, self.n_hidden)
                Wb_3 = self.quantize(W_3)
                self.w3_summ = tf.summary.histogram(name='W3_summ', values=W_3)
                self.wb3_summ = tf.summary.histogram(
                    name='Wb3_summ', values=Wb_3)

                # don't quantize input (fc3) to last layer (fcout_b)
                if self.fast:
                    fc3 = tf.nn.dropout(xnor_gemm(fc2, Wb_3), self.keep_prob)
                else:
                    fc3 = tf.nn.dropout(tf.matmul(fc2, Wb_3), self.keep_prob)
                self.fc3_summ = tf.summary.histogram(
                    name='a3_summ', values=fc3)

            with tf.name_scope('fcout_b') as scope:

                W_out = self.init_layer('W_out', self.n_hidden, 10)
                self.wout_summ = tf.summary.histogram(
                    name='Wout_summ', values=W_out)

                self.output = tf.matmul(fc3, W_out)
                self.aout_summ = tf.summary.histogram(
                    name='aout_summ', values=self.output)
        else:

            with tf.name_scope('fc1_fp') as scope:

                W_1 = self.init_layer('W_1', 784, self.n_hidden)
                self.w1_summ = tf.summary.histogram(name='W1_summ', values=W_1)
                fc1 = tf.nn.dropout(tf.nn.relu(tf.matmul(self.input, W_1)), self.keep_prob)
                if batch_norm:
                    fc1 = tf.contrib.layers.batch_norm(
                        fc1, center=True, scale=True, epsilon=BN_EPSILON, is_training=phase)

            with tf.name_scope('fc2_fp') as scope:

                W_2 = self.init_layer('W_2', self.n_hidden, self.n_hidden)
                self.w2_summ = tf.summary.histogram(name='W2_summ', values=W_2)
                fc2 = tf.nn.dropout(tf.nn.relu(tf.matmul(fc1, W_2)), self.keep_prob)
                if batch_norm:
                    fc2 = tf.contrib.layers.batch_norm(
                        fc2, center=True, scale=True, epsilon=BN_EPSILON, is_training=phase)

            with tf.name_scope('fc3_fp') as scope:

                W_3 = self.init_layer('W_3', self.n_hidden, self.n_hidden)
                self.w3_summ = tf.summary.histogram(name='W3_summ', values=W_3)
                fc3 = tf.nn.dropout(tf.nn.relu(tf.matmul(fc2, W_3)), self.keep_prob)
                if batch_norm:
                    fc3 = tf.contrib.layers.batch_norm(
                        fc3, center=True, scale=True, epsilon=BN_EPSILON, is_training=phase)

            with tf.name_scope('fcout_fp') as scope:

                W_out = self.init_layer('W_out', self.n_hidden, 10)
                self.wout_summ = tf.summary.histogram(
                    name='Wout_summ', values=W_out)
                self.output = tf.matmul(fc3, W_out)
                if batch_norm:
                    self.output = tf.contrib.layers.batch_norm(
                        self.output, center=True, scale=True, epsilon=BN_EPSILON, is_training=phase)
