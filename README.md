# tensorflow-xnor-bnn
BinaryNets in TensorFlow with XNOR GEMM op

### Dependencies
The project was tested with:
* `python 3.6.1`
* `tensorflow 1.2.1`
* `numpy 1.13.1`
* `g++ 4.8.4`
* `Cuda compilation tools, release 8.0, V8.0.44`

## Using this repo

### 1 - Compile the gemm_op.so library
1. Run `source setenv.sh` to set `TF_INC` variable with location to core tensorflow headers (you do not need to have source installed). 
2. In project root run `mkdir libs`, this is where `gemm_op.so` will be placed.
3. Run `make`. If you want to make changes to the op without changing the kernels, there is a `cpp` target to save time. 

### 2 - Confirm the op yields same results as tf.matmul()
1. Run `python test_gemm_op.py` which generates two matrices of +1/-1 and compares the results from `xnor_gemm` to `tf.matmul`.

### 3 - Run benchmarks
1. Run `python matmul_bench.py` to compare the GEMM performance between the `xnor_gemm` and `tf.matmul`. The speedup is less than that reported in https://arxiv.org/abs/1602.02830 because we're comparing to a highly optimized kernel, not the unoptimized base kernel. The results should be similar to the improvement over cuBLAS (2-3x for large matrices).

GTX-680-4GB

|N     | RUNS| Avg	(s)| Std	(s)| Avg (s) | Std	(s)| Speedup |
|------|:----|:--------|:--------|:--------|:--------|:-------:|
|1024	 |20	 |0.00608	 |0.00051	 |0.00875	 |0.01861	 |1.44     |
|2048	 |10	 |0.01877	 |0.00235	 |0.02770	 |0.02294	 |1.48     |
|4096	 |10	 |0.07897	 |0.00325	 |0.11908	 |0.02427	 |1.51     |
|8192	 |10	 |0.36292	 |0.00331	 |0.75703	 |0.02268	 |2.09     |

GTX-TITAN-BLACK-6GB

|N | RUNS	| Avg	(s) | Std	(s) | Avg (s)	| Std	(s) | Speedup |
|------|:----|:--------|:--------|:--------|:--------|:----:|
| 1024 | 20  | 0.00473 | 0.00021 | 0.00362 | 0.00199 | 0.76 |
| 2048 | 10  | 0.01184 | 0.00007 | 0.01364 | 0.00879 | 1.15 |
| 4096 | 10  | 0.04598 | 0.00524 | 0.06320 | 0.01995 | 1.37 |
| 8192 | 10  | 0.19189 | 0.00323 | 0.35513 | 0.08722 | 1.85 |

TESLA-P100-PCIE-12GB

| N    | RUNS | Avg (s) | Med (s)  | Avg (s) | Med (s) | Speedup (avg) | Speedup (med) |
|------|------|---------|----------|---------|---------|---------------|---------------|
| 1024 | 19   | 0.00316 | 0.00317  | 0.00264 | 0.00202 | 0.83          | 0.64          |
| 2048 | 9    | 0.00804 | 0.008029 | 0.01028 | 0.00698 | 1.28          | 0.87          |
| 4096 | 9    | 0.02665 | 0.02647  | 0.04669 | 0.03353 | 1.75          | 1.27          |
| 8192 | 9    | 0.10526 | 0.10534  | 0.23801 | 0.19075 | 2.26          | 1.81          |


## Limitations
XNOR GEMM op currently only works for square matrices that are powers of 2, with smallest N being 512.

## Relevant links
- https://github.com/tensorflow/tensorflow/issues/1592
- https://www.tensorflow.org/extend/adding_an_op
