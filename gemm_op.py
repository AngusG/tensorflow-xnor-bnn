import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

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
    '''
    to_zero = op.inputs[0]
    shape = array_ops.shape(to_zero)
    index = array_ops.zeros_like(shape)
    first_grad = array_ops.reshape(grad, [-1])[0]
    to_zero_grad = sparse_ops.sparse_to_dense([index], shape, first_grad, 0)
    '''

    

    return [to_zero_grad]  # List of one Tensor, since we have one input
