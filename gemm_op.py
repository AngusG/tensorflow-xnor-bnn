import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

gemm_module = tf.load_op_library('./libs/gemm_op.so')
xnor_gemm = gemm_module.gemm


@ops.RegisterGradient("Gemm")
def _xnor_gemm_grad(op, grad):
    """The gradients for `xnor_gemm`.

    Args:
      op: The `xnor_gemm` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
      grad: Gradient with respect to the output of the `xnor_gemm` op.

    Returns:
      Gradients with respect to the input of `xnor_gemm`.
    """
    a = op.inputs[0]
    b = op.inputs[1]
    grad_a = math_ops.matmul(grad, b, transpose_b=True)
    grad_b = math_ops.matmul(a, grad, transpose_a=True)
    return grad_a, grad_b
