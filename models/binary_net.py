import tensorflow as tf
from tf_gemm_op import xnor_gemm

BN_EPSILON = 1e-4


@tf.RegisterGradient("QuantizeGrad")
def quantize_grad(op, grad):
    return tf.clip_by_value(tf.identity(grad), -1, 1)


class BinaryNet:

    def __init__(self, binary, first, last, xnor, n_hidden, keep_prob, x, batch_norm, phase):
        self.binary = binary
        self.xnor = xnor
        self.n_hidden = n_hidden
        self.keep_prob = keep_prob
        self.input = x
        self.G = tf.get_default_graph()
        self.dense_layers(batch_norm, first, last, phase)

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

    def dense_layers(self, batch_norm, first, last, phase):

        if self.binary:

            with tf.name_scope('fc1_b') as scope:

                W_1 = self.init_layer('W_1', 784, self.n_hidden)
                self.w1_summ = tf.summary.histogram(name='W1_summ', values=W_1)

                # optionally quantize weights in first layer
                if first:
                    self.W_1_p = tf.reduce_sum(1.0 - tf.square(W_1))
                    Wb_1 = self.quantize(W_1)
                    fc1 = tf.nn.dropout(tf.matmul(self.input, Wb_1), self.keep_prob)
                else:
                    fc1 = tf.nn.dropout(tf.matmul(self.input, W_1), self.keep_prob)

                if batch_norm:
                    fc1 = tf.contrib.layers.batch_norm(
                        fc1, decay=0.9, center=False, scale=False, epsilon=BN_EPSILON, is_training=phase)

                if first:
                    fc1_scale_factor = tf.Variable(0.05, trainable=True, name='fc1_scale_factor')
                    self.fc1_scale_factor_summary = tf.summary.scalar("fc1 scale factor", fc1_scale_factor)
                    fc1 = fc1 * fc1_scale_factor
                
                self.a1_fp_summ = tf.summary.histogram(
                        name='a1_fp_summ', values=fc1)
                fc1 = self.quantize(fc1)
                self.a1_bin_summ = tf.summary.histogram(
                    name='a1_bin_summ', values=fc1)

            with tf.name_scope('fc2_b') as scope:

                W_2 = self.init_layer('W_2', self.n_hidden, self.n_hidden)
                self.W_2_p = tf.reduce_sum(1.0 - tf.square(W_2))

                Wb_2 = self.quantize(W_2)
                self.w2_summ = tf.summary.histogram(name='W2_summ', values=W_2)
                self.wb2_summ = tf.summary.histogram(
                    name='Wb2_summ', values=Wb_2)

                if self.xnor:
                    fc2 = tf.nn.dropout(xnor_gemm(fc1, Wb_2), self.keep_prob)
                else:
                    fc2 = tf.nn.dropout(tf.matmul(fc1, Wb_2), self.keep_prob)

                if batch_norm:
                    fc2 = tf.contrib.layers.batch_norm(
                        fc2, decay=0.9, center=False, scale=False, epsilon=BN_EPSILON, is_training=phase)
                self.a2_fp_summ = tf.summary.histogram(
                    name='a2_fp_summ', values=fc2)
                fc2 = self.quantize(fc2)
                self.a2_bin_summ = tf.summary.histogram(
                    name='a2_bin_summ', values=fc2)

            with tf.name_scope('fc3_b') as scope:

                W_3 = self.init_layer('W_3', self.n_hidden, self.n_hidden)
                self.W_3_p = tf.reduce_sum(1.0 - tf.square(W_3))
                Wb_3 = self.quantize(W_3)
                self.w3_summ = tf.summary.histogram(name='W3_summ', values=W_3)
                self.wb3_summ = tf.summary.histogram(
                    name='Wb3_summ', values=Wb_3)

                if self.xnor:
                    fc3 = tf.nn.dropout(xnor_gemm(fc2, Wb_3), self.keep_prob)
                else:
                    fc3 = tf.nn.dropout(tf.matmul(fc2, Wb_3), self.keep_prob)

                if batch_norm:
                    fc3 = tf.contrib.layers.batch_norm(
                        fc3, decay=0.9, center=False, scale=False, epsilon=BN_EPSILON, is_training=phase)
                
                # only quantize input to last layer if received 'last' flag
                if last:
                    fc3 = self.quantize(fc3)
                    name='a3_bin_summ'
                else:
                    name='a3_fp_summ'

                self.a3_summ = tf.summary.histogram(name=name, values=fc3)

            with tf.name_scope('fcout_b') as scope:

                W_out = self.init_layer('W_out', self.n_hidden, 10)
                self.wout_summ = tf.summary.histogram(
                    name='Wout_summ', values=W_out)

                if last:
                    Wb_out = self.quantize(W_out)
                    self.W_out_p = tf.reduce_sum(1.0 - tf.square(W_out))
                    self.wbout_summ = tf.summary.histogram(
                        name='Wbout_summ', values=Wb_out)

                    '''
                    Output scale factor from Tang et al. 2017, initialized to 0.05 
                    (instead of 0.0001) as activations grow by ~10x when binarizing 
                    W_out and fc3. In experiments the scale factor tends to stabilize around 0.05.
                    '''
                    out_scale_factor = tf.Variable(0.05, trainable=True, name='out_scale_factor')
                    self.out_scale_factor_summary = tf.summary.scalar("out scale factor", out_scale_factor)
                    self.output = tf.matmul(fc3, Wb_out) * out_scale_factor
                else:
                    self.output = tf.matmul(fc3, W_out)

                self.aout_summ = tf.summary.histogram(
                    name='aout_summ', values=self.output)
        # fp32 net                
        else:

            with tf.name_scope('fc1_fp') as scope:

                W_1 = self.init_layer('W_1', 784, self.n_hidden)
                self.w1_summ = tf.summary.histogram(name='W1_summ', values=W_1)
                fc1 = tf.matmul(self.input, W_1)
                if batch_norm:
                    fc1 = tf.contrib.layers.batch_norm(
                        fc1, center=False, scale=False, epsilon=BN_EPSILON, is_training=phase)
                fc1 = tf.nn.dropout(tf.nn.relu(fc1), self.keep_prob)
                self.a1_fp_summ = tf.summary.histogram(
                        name='a1_fp_summ', values=fc1)

            with tf.name_scope('fc2_fp') as scope:

                W_2 = self.init_layer('W_2', self.n_hidden, self.n_hidden)
                self.w2_summ = tf.summary.histogram(name='W2_summ', values=W_2)
                fc2 = tf.matmul(fc1, W_2)
                if batch_norm:
                    fc2 = tf.contrib.layers.batch_norm(
                        fc2, center=False, scale=False, epsilon=BN_EPSILON, is_training=phase)
                fc2 = tf.nn.dropout(tf.nn.relu(fc2), self.keep_prob)
                self.a2_fp_summ = tf.summary.histogram(
                        name='a2_fp_summ', values=fc2)

            with tf.name_scope('fc3_fp') as scope:

                W_3 = self.init_layer('W_3', self.n_hidden, self.n_hidden)
                self.w3_summ = tf.summary.histogram(name='W3_summ', values=W_3)
                fc3 = tf.matmul(fc2, W_3)
                if batch_norm:
                    fc3 = tf.contrib.layers.batch_norm(
                        fc3, center=False, scale=False, epsilon=BN_EPSILON, is_training=phase)
                fc3 = tf.nn.dropout(tf.nn.relu(fc3), self.keep_prob)
                self.a3_fp_summ = tf.summary.histogram(
                        name='a3_fp_summ', values=fc3)

            with tf.name_scope('fcout_fp') as scope:

                W_out = self.init_layer('W_out', self.n_hidden, 10)
                self.wout_summ = tf.summary.histogram(
                    name='Wout_summ', values=W_out)
                self.output = tf.matmul(fc3, W_out)
                self.aout_summ = tf.summary.histogram(
                    name='aout_summ', values=self.output)
                '''
                if batch_norm:
                    self.output = tf.contrib.layers.batch_norm(
                        self.output, center=False, scale=False, epsilon=BN_EPSILON, is_training=phase)
                '''
