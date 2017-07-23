import tensorflow as tf
from gemm_op import xnor_gemm


class GemmTest(tf.test.TestCase):

    def testGemm(self):
        with self.test_session():

            N = 512

            a = tf.sign(tf.random_normal(shape=[N, N], seed=1).eval())
            b = tf.sign(tf.random_normal(shape=[N, N], seed=2).eval())

            xnor_result = xnor_gemm(a, b)
            tf_result = tf.matmul(a, b)

            print("Result for xnor_gemm()\n")
            print(xnor_result.eval())
            print("Result for tf.matmul()\n")
            print(tf_result.eval())

            self.assertAllEqual(xnor_result.eval(), tf_result.eval())

if __name__ == "__main__":
    tf.test.main()
