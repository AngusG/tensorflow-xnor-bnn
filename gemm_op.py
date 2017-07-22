import tensorflow as tf

gemm_module = tf.load_op_library('./libs/gemm_op.so')

xnor_gemm = gemm_module.gemm