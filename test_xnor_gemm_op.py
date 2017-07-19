import tensorflow as tf


class XnorGemmTest(tf.test.TestCase):

    def testXnorGemm(self):
        xnor_gemm_module = tf.load_op_library('./libs/xnor_gemm_kernel.so')
        with self.test_session():

        	N=32

            a = tf.constant([1, -1, 1, -1, 1, -1], shape=[2, 3])
            b = tf.constant([-1, 1, -1, 1, -1, 1], shape=[3, 2])
            '''
            a = tf.random_uniform(
                shape=[32, 32], minval=0, maxval=2, dtype=tf.int32, seed=1)
            b = tf.random_uniform(
                shape=[32, 32], minval=0, maxval=2, dtype=tf.int32, seed=1)
			'''                
			a = tf.cast(2 * (tf.random_normal(shape=[N,N],seed=1).eval() > 0) - 1, tf.int32)
			b = tf.cast(2 * (tf.random_normal(shape=[N,N],seed=2).eval() > 0) - 1, tf.int32)

            xnor_result = xnor_gemm_module.xnor_gemm(a, b)
            base_result = tf.matmul(a, b)

            #self.assertAllEqual(result.eval(), [0, 2, 0, 1], shape=[2,2])
            self.assertAllEqual(xnor_result.eval(), base_result.eval())

if __name__ == "__main__":
    tf.test.main()
