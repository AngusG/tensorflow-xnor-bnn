// xnor_gemm_kernel.h
#ifndef XNOR_GEMM_KERNEL_H_
#define XNOR_GEMM_KERNEL_H_

template <typename Device, typename T>
struct XnorGemmFunctor {
  // Computes on device "d": C = A * B, where * is matrix multiplication.
  void operator()(const Device& d, const int* A, const int* B, float* C, const int m, const int n, const int k);
};

#endif // XNOR_GEMM_KERNEL_H_
