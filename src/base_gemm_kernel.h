// base_gemm_kernel.h
#ifndef BASE_GEMM_KERNEL_H_
#define BASE_GEMM_KERNEL_H_

template <typename Device, typename T>
struct BaseGemmFunctor {
  // Computes on device "d": C = A * B, where * is matrix multiplication.
  void operator()(const Device& d, const float* A, const float* B, float* C, const int m, const int n, const int k);
};

#endif // BASE_GEMM_KERNEL_H_
