import time
import numpy as np
import tensorflow as tf
from gemm_op import xnor_gemm

N = 1024
N_RUNS = 20

print("Running for %d runs with size %d" % (N_RUNS,N))

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

    for i in range(N_RUNS):
        ########### benchmark matmul ##########
        start_time = time.time()
        matmul_result = sess.run(matmul, feed_dict={A: a, B: a})
        #matmul_result = sess.run(tf.matmul(a_f32, a_f32))
        base_timings[i] = time.time() - start_time

        print("matmul %d took %f" % (i, base_timings[i]))
        print(matmul_result)
        #######################################

    print("Avg XNOR   execution time over %d runs: %f +/- %f" % (N_RUNS-1, xnor_timings[1:].mean(), xnor_timings[1:].std()))
    print("Avg MatMul execution time over %d runs: %f +/- %f" % (N_RUNS-1, base_timings[1:].mean(), base_timings[1:].std()))

    print("Med XNOR   execution time over %d runs: %f +/- %f" % (N_RUNS-1, np.median(xnor_timings[1:]), xnor_timings[1:].std()))
    print("Med MatMul execution time over %d runs: %f +/- %f" % (N_RUNS-1, np.median(base_timings[1:]), base_timings[1:].std()))
