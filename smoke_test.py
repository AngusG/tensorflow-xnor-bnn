import time
import numpy as np
import tensorflow as tf
from gemm_op import xnor_gemm

N = 8196
N_RUNS = 5

A = tf.placeholder(tf.float32, [N, N])
B = tf.placeholder(tf.float32, [N, N])
xnor_gemm = xnor_gemm(A, B)
matmul = tf.matmul(A, B)

# Re-use a for benchmarking on GPU w/only 4GB memory
a_T = tf.sign(tf.random_normal(shape=[N, N], seed=1))
b_T = tf.sign(tf.random_normal(shape=[N, N], seed=2))

xnor_timings = np.zeros(N_RUNS)
base_timings = np.zeros(N_RUNS)

with tf.Session() as sess:

    a = sess.run(a_T)
    b = sess.run(b_T)
    
    for i in range(N_RUNS):
        ########### benchmark xnor ############
        start_time = time.time()
        xnor_gemm_result = sess.run(xnor_gemm, feed_dict={A: a, B: a})
        #xnor_gemm_result = sess.run(xnor_gemm(a_f32, a_f32))
        xnor_timings[i] = time.time() - start_time

        print("xnor_gemm %d took %f" % (i, xnor_timings[i]))
        print(xnor_gemm_result)
        #######################################
    print("Avg XNOR kernel execution time over %d runs: %f +/- %f" % (N_RUNS, xnor_timings.mean(), xnor_timings.std()))

    for i in range(N_RUNS):
        ########### benchmark matmul ##########
        start_time = time.time()
        matmul_result = sess.run(matmul, feed_dict={A: a, B: a})
        #matmul_result = sess.run(tf.matmul(a_f32, a_f32))
        base_timings[i] = time.time() - start_time

        print("matmul %d took %f" % (i, base_timings[i]))
        print(matmul_result)
        #######################################
    print("Avg MatMul execution time over %d runs: %f +/- %f" % (N_RUNS, base_timings.mean(), base_timings.std()))
