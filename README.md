# tensorflow-xnor-bnn
BinaryNets in TensorFlow with XNOR GEMM op

## Building
First do: 
`source setenv.sh`
then 
`make`

## Limitations
XNOR GEMM op currently only works for square matrices that are powers of 2, with smallest N being 512.

## Relevant links
- https://github.com/tensorflow/tensorflow/issues/1592
- https://www.tensorflow.org/extend/adding_an_op
