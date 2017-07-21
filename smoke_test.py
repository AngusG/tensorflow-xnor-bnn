import time
import tensorflow as tf

N = 8192

gemm_module = tf.load_op_library('./libs/gemm_op.so')

A = tf.placeholder(tf.float32, [N, N])
B = tf.placeholder(tf.float32, [N, N])

a = 2 * tf.cast(tf.random_normal(shape=[N, N], seed=1) > 0, tf.float32) - 1
b = 2 * tf.cast(tf.random_normal(shape=[N, N], seed=2) > 0, tf.float32) - 1

xnor_gemm = gemm_module.gemm(A, B)
matmul = tf.matmul(a, b)

with tf.Session() as sess:

    a_f32 = sess.run(a)
    b_f32 = sess.run(b)

    ########### benchmark xnor ############
    start_time = time.time()
    xnor_gemm_result = sess.run(xnor_gemm, feed_dict={A: a_f32, B: b_f32})
    xnor_gemm_time = time.time() - start_time

    print("xnor_gemm took %f" % xnor_time)
    print(xnor_gemm_result)
    #######################################

    ########### benchmark matmul ##########
    start_time = time.time()
    matmul_result = sess.run(matmul, feed_dict={A: a_f32, B: b_f32})
    matmul_time = time.time() - start_time
    
    print("matmul took %f" % tf_time)
    print(matmul_result)
    #######################################
