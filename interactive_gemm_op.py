import time
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

    a_float = tf.cast(
        2 * (tf.random_normal(shape=[N, N], seed=1).eval() > 0) - 1, tf.float32)
    b_float = tf.cast(
        2 * (tf.random_normal(shape=[N, N], seed=2).eval() > 0) - 1, tf.float32)

    start_time = time.time()
    xnor_result = gemm_module.gemm(a_float, b_float)
    xnor_result.eval()
    xnor_time = time.time() - start_time

    start_time = time.time()
    tf_result = tf.matmul(a_float, b_float)
    tf_result.eval()
    tf_time = time.time() - start_time

    print(xnor_result.eval())
    print(tf_result.eval())

    print("xnor_gemm() took %f" % xnor_time)
    print("tf.matmul() took %f" % tf_time)
