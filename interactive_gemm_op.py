import argparse
import tensorflow as tf

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n', help='matrix dimension (square)', type=int, default=512)
    parser.add_argument(
        '--runs', help='how many times to run to avg random error', type=int, default=10)
    args = parser.parse_args()

    N = args.n

    gemm_module = tf.load_op_library('./libs/gemm_op.so')
    sess = tf.InteractiveSession()

    # Generate matrices with values constrained to -1, 1
    '''
    a_int = tf.cast(2 * (tf.random_normal(shape=[N,N],seed=1).eval() > 0) - 1, tf.int32)
    b_int = tf.cast(2 * (tf.random_normal(shape=[N,N],seed=2).eval() > 0) - 1, tf.int32)
	# not working yet
    xnor_result = gemm_module.gemm(a_float, b_float)
    '''
    a_float = tf.cast(2 * (tf.random_normal(shape=[N,N],seed=1).eval() > 0) - 1, tf.float32)
    b_float = tf.cast(2 * (tf.random_normal(shape=[N,N],seed=2).eval() > 0) - 1, tf.float32)

    print(gemm_module.gemm(a_float, b_float).eval())
    print(tf.matmul(a_float, b_float).eval())
