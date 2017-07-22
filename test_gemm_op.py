import tensorflow as tf
from gemm_op import xnor_gemm


class GemmTest(tf.test.TestCase):

    def testGemm(self):
        #gemm_module = tf.load_op_library('./libs/gemm_op.so')
        with self.test_session():

            N = 512

            a_float = tf.cast(2 * (tf.random_normal(shape=[N,N],seed=1).eval() > 0) - 1, tf.float32)
            b_float = tf.cast(2 * (tf.random_normal(shape=[N,N],seed=2).eval() > 0) - 1, tf.float32)

            #xnor_result = gemm_module.gemm(a_float, b_float)
            xnor_result = xnor_gemm(a_float, b_float)
            tf_result = tf.matmul(a_float, b_float)

            print("Result for xnor_gemm()\n")
            print(xnor_result.eval())
            print("Result for tf.matmul()\n")
            print(tf_result.eval())

            self.assertAllEqual(xnor_result.eval(), tf_result.eval())

if __name__ == "__main__":
    tf.test.main()
