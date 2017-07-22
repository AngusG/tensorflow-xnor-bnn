import tensorflow as tf
from gemm_op import xnor_gemm


class GemmTest(tf.test.TestCase):

    def testGemm(self):
        #gemm_module = tf.load_op_library('./libs/gemm_op.so')
        with self.test_session():

            N = 4096

            a = tf.sign(tf.random_normal(shape=[N, N], seed=1))
            #b = tf.sign(tf.random_normal(shape=[N, N], seed=2))

            #xnor_result = gemm_module.gemm(a, b)
            xnor_result = xnor_gemm(a, a)
            tf_result = tf.matmul(a, a)

            print("Result for xnor_gemm()\n")
            print(xnor_result.eval())
            print("Result for tf.matmul()\n")
            print(tf_result.eval())

            self.assertAllEqual(xnor_result.eval(), tf_result.eval())

if __name__ == "__main__":
    tf.test.main()
