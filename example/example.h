// example.h
#ifndef KERNEL_EXAMPLE_H_
#define KERNEL_EXAMPLE_H_

template <typename Device, typename T>
struct ExampleFunctor {
  void operator()(const Device& d, int size, const T* in, T* out);
};

#endif // KERNEL_EXAMPLE_H_
