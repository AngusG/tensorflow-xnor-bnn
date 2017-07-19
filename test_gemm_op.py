import tensorflow as tf


class GemmTest(tf.test.TestCase):

    def testGemm(self):
        gemm_module = tf.load_op_library('./libs/gemm_op.so')
        with self.test_session():

            N = 512

            # Generate matrices with values constrained to -1, 1
            #a_int = tf.cast(2 * (tf.random_normal(shape=[N,N],seed=1).eval() > 0) - 1, tf.int32)
            #b_int = tf.cast(2 * (tf.random_normal(shape=[N,N],seed=2).eval() > 0) - 1, tf.int32)

            a_float = tf.cast(2 * (tf.random_normal(shape=[N,N],seed=1).eval() > 0) - 1, tf.float32)
            b_float = tf.cast(2 * (tf.random_normal(shape=[N,N],seed=2).eval() > 0) - 1, tf.float32)

            base_result = gemm_module.gemm(a_float, b_float)
            tf_result = tf.matmul(a_float, b_float)

            self.assertAllEqual(base_result.eval(),tf_result.eval())

if __name__ == "__main__":
    tf.test.main()
