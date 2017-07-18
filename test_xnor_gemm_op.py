import tensorflow as tf


class XnorGemmTest(tf.test.TestCase):

    def testXnorGemm(self):
        xnor_gemm_module = tf.load_op_library('./libs/xnor_gemm_kernel.so')
        with self.test_session():
        	
        	a = tf.constant([1, 0, 1, 0, 1, 0], shape=[2, 3])
        	b = tf.constant([0, 1, 0, 1, 0, 1], shape=[3, 2])

            result = xnor_gemm_module.xnor_gemm(a,b)
            self.assertAllEqual(result.eval(), [0, 2, 0, 1], shape=[2,2])

if __name__ == "__main__":
    tf.test.main()
