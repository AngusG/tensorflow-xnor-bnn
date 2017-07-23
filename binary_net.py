import tensorflow as tf
from gemm_op import xnor_gemm

'''
# use tf.sign() instead
# weight and activation binarization function -- eq.(1) in Courbariaux et al. 
def sign(x):
    return 2 * tf.cast(x > 0, tf.float32) - 1
'''

# hard sigmoid -- eq.(3) in Courbariaux et al.


#def binary_sigmoid_unit(x):
#    return hard_sigmoid(x)    

'''
# Activation binarization function
def SignTheano(x):
    return tf.subtract(tf.multiply(tf.cast(tf.greater_equal(x, tf.zeros(tf.shape(x))), tf.float32), 2.0), 1.0)
'''

# The weights' binarization function,
# taken directly from the BinaryConnect github repository and simplified
# (which was made available by his authors)
'''
def binarization(W, H, binary=True):

    if not binary:
        Wb = W
    else:
        # [-1,1] -> [0,1]
        Wb = hard_sigmoid(W / H)
        Wb = tf.round(Wb)

        # 0 or 1 -> -1 or 1
        Wb = tf.cast(tf.where(Wb, H, -H), tf.float32)
    return Wb
'''    

class BinaryNet:    

    def __init__(self, n_hidden, x):
        self.n_hidden = n_hidden
        self.input = x
        self.G = tf.get_default_graph()
        self.dense_layers()

    def init_layer(self, name, n_inputs, n_outputs):
        
        W = tf.get_variable(name, shape=[n_inputs, n_outputs], initializer=tf.contrib.layers.xavier_initializer())
        #b = tf.Variable(tf.zeros([n_outputs]))
        
        return W

    '''
    def hard_sigmoid(self, x):
        return tf.clip_by_value((x + 1.) / 2, 0, 1)

    def binary_tanh_unit(self, x):
        return 2 * self.hard_sigmoid(x) - 1

    def binarize(self, W):

        # [-1,1] -> [0,1]
        #Wb = tf.round(self.hard_sigmoid(W))
        Wb = self.hard_sigmoid(W)
        plus_one = tf.ones_like(Wb)
        neg_one = -1 * tf.ones_like(Wb)
        # 0 or 1 -> -1 or 1
        Wb = tf.cast(tf.where(tf.cast(Wb, tf.bool), plus_one, neg_one), tf.float32)
        return Wb            

    def dense_layers(self):

        with tf.name_scope('fc1_fp') as scope:

            W_1 = self.init_layer('W_1', 784, self.n_hidden)
            fc_1 = self.binarize(tf.nn.relu(tf.matmul(self.input, self.binarize(W_1))))

        with tf.name_scope('fc2_xnor') as scope:

            W_2 = self.init_layer('W_2', self.n_hidden, self.n_hidden)
            fc_2 = self.binarize(tf.nn.relu(xnor_gemm(fc_1, self.binarize(W_2))))

        with tf.name_scope('fc3_fp') as scope:

            W_3 = self.init_layer('W_3', self.n_hidden, 10)
            self.output = tf.matmul(fc_2, self.binarize(W_3))
    '''            

    
    def quantize(self, x): 
        with self.G.gradient_override_map({"Sign": "Identity"}):
        	return tf.sign(x)
            #E = tf.reduce_mean(tf.abs(x))
            #return tf.sign(x) * E

    def dense_layers(self):

        with tf.name_scope('fc1_fp') as scope:

            W_1 = self.init_layer('W_1', 784, self.n_hidden)
            fc_1 = self.quantize(tf.nn.relu(tf.matmul(self.input, self.quantize(W_1))))

        with tf.name_scope('fc2_xnor') as scope:

            W_2 = self.init_layer('W_2', self.n_hidden, self.n_hidden)
            fc_2 = self.quantize(tf.nn.relu(xnor_gemm(fc_1, self.quantize(W_2))))

        with tf.name_scope('fc3_fp') as scope:

            W_3 = self.init_layer('W_3', self.n_hidden, 10)
            self.output = tf.matmul(fc_2, self.quantize(W_3))