// xnor_gemm_kernel.h
#ifndef XNOR_GEMM_KERNEL_H_
#define XNOR_GEMM_KERNEL_H_

template <typename Device, typename T>
struct XnorGemmFunctor {
  // Computes on device "d": C = A * B, where * is matrix multiplication.
  //void operator()(const Device& d, const T* A, const T* B, T* C, int m, int n, int k);
  //void operator()(const Device& d, unsigned int* A, unsigned int* B, float* C, const T* m, const T* n, const T* k);
  void operator()(const Device& d, const int* A, const int* B, float* C, const int m, const int n, const int k);
  	  /*
      const Device& d, typename MatMulTypes<T>::out_type out,
      typename MatMulTypes<T>::in_type in0,
      typename MatMulTypes<T>::in_type in1,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair
      #endif);
      */
};

#endif // XNOR_GEMM_KERNEL_H_
