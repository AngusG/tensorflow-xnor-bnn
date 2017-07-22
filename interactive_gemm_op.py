import time
import numpy as np
import tensorflow as tf
from gemm_op import xnor_gemm

N = 4096
N_RUNS = 5

sess = tf.InteractiveSession()

a = tf.cast(
    2 * (tf.random_normal(shape=[N, N], seed=1).eval() > 0) - 1, tf.float32)
# b = tf.cast(
#    2 * (tf.random_normal(shape=[N, N], seed=2).eval() > 0) - 1, tf.float32)

xnor_timings = np.zeros(N_RUNS)
base_timings = np.zeros(N_RUNS)

for i in range(N_RUNS):
    start_time = time.time()
    xnor_gemm(a, a).eval()
    xnor_timings[i] = time.time() - start_time
    print("xnor_gemm %d took %f" % (i, xnor_timings[i]))
print("Avg XNOR kernel execution time over %d runs: %f +/- %f" % (N_RUNS - 1,
                                                                  xnor_timings[1:].mean(), xnor_timings[1:].std()))
for i in range(N_RUNS):
    start_time = time.time()
    tf.matmul(a, a).eval()
    base_timings[i] = time.time() - start_time
    print("matmul %d took %f" % (i, base_timings[i]))
print("Avg MatMul execution time over %d runs: %f +/- %f" % (N_RUNS - 1,
                                                             base_timings[1:].mean(), base_timings[1:].std()))