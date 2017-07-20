// concatenate_kernel.h
#ifndef CONCATENATE_KERNEL_H_
#define CONCATENATE_KERNEL_H_

template <typename Device, typename T>
struct ConcatenateRowsFunctor {
  void operator()(const Device& d, const float* fA, int* Aconc, const int N);
};

template <typename Device, typename T>
struct ConcatenateColsFunctor {
  void operator()(const Device& d, const float* fB, int* Bconc, const int N);
};

#endif // CONCATENATE_KERNEL_H_
