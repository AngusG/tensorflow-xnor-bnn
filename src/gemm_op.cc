// gemm_op.cc
#define EIGEN_USE_THREADS

#include <stdio.h>
#include "xnor_gemm_kernel.h"
#include "base_gemm_kernel.h"
#include "concatenate_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
//#include "tensorflow/core/kernels/fill_functor.h"
//#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

//typedef Eigen::ThreadPoolDevice CPUDevice;
//typedef Eigen::GpuDevice GPUDevice;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

//TODO: CPU specialization of xnor_gemm computation.
/*
template <typename T>
struct XnorGemmFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, const T* A, const T* B, T* C, int m, int n, const uint64 k) {
    for (int i = 0; i < size; ++i) {
      out[i] = 2 * in[i];
  }
};
*/

template <typename Device, typename T>
class BaseGemmOp : public OpKernel {
 public:
  explicit BaseGemmOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  /* Compute method must be thread-safe */
  void Compute(OpKernelContext* ctx) override {
    
    // Get the input tensors
    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);

    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(a.shape()),
                errors::InvalidArgument("In[0] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(b.shape()),
                errors::InvalidArgument("In[1] is not a matrix"));
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
    dim_pair[0].first = transpose_a_ ? 0 : 1;
    dim_pair[0].second = transpose_b_ ? 1 : 0;

    OP_REQUIRES(
        ctx, a.dim_size(dim_pair[0].first) == b.dim_size(dim_pair[0].second),
        errors::InvalidArgument(
            "Matrix size-incompatible: In[0]: ", a.shape().DebugString(),
            ", In[1]: ", b.shape().DebugString()));
    int a_dim_remaining = 1 - dim_pair[0].first;
    int b_dim_remaining = 1 - dim_pair[0].second;
    TensorShape out_shape(
        {a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));

    if (out->NumElements() == 0) {
      // If a has shape [0, x] or b has shape [x, 0], the output shape
      // is a 0-element matrix, so there is nothing to do.
      return;
    }

    if (a.NumElements() == 0 || b.NumElements() == 0) {
      // If a has shape [x, 0] and b has shape [0, y], the
      // output shape is [x, y] where x and y are non-zero, so we fill
      // the output with zeros.
      
      /*
      functor::SetZeroFunctor<Device, T> f;
      f(ctx->eigen_device<Device>(), out->flat<T>());
      */
      return;
    }

    const int32 m = a.dim_size(1 - dim_pair[0].first);
    const int32 k = a.dim_size(dim_pair[0].first);
    const int32 n = b.dim_size(1 - dim_pair[0].second);

    printf("m = %d ", m);
    printf("k = %d ", k);
    printf("n = %d ", n);

    BaseGemmFunctor<Device, T>()(
    ctx->eigen_device<Device>(),
    a.flat<T>().data(), // data is const here, which is why const was added to kernels
    b.flat<T>().data(),
    out->flat<float>().data(),
    m, 
    n,
    k);
  }
  private:
  bool transpose_a_;
  bool transpose_b_;
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
//template <typename Device>
template <typename Device, typename T>
class XnorGemmOp : public OpKernel {
 public:
  explicit XnorGemmOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  /* Compute method must be thread-safe */
  void Compute(OpKernelContext* ctx) override {
    
    // Get the input tensors
    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);

    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(a.shape()),
                errors::InvalidArgument("In[0] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(b.shape()),
                errors::InvalidArgument("In[1] is not a matrix"));
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
    dim_pair[0].first = transpose_a_ ? 0 : 1;
    dim_pair[0].second = transpose_b_ ? 1 : 0;

    OP_REQUIRES(
        ctx, a.dim_size(dim_pair[0].first) == b.dim_size(dim_pair[0].second),
        errors::InvalidArgument(
            "Matrix size-incompatible: In[0]: ", a.shape().DebugString(),
            ", In[1]: ", b.shape().DebugString()));
    int a_dim_remaining = 1 - dim_pair[0].first;
    int b_dim_remaining = 1 - dim_pair[0].second;
    
    TensorShape out_shape(
        {a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)});
    Tensor* out = nullptr;

    /* From: core/framework/op_kernel.h */
    /* Status allocate_output(int index, const TensorShape& shape,
                         Tensor** tensor) TF_MUST_USE_RESULT;
    */
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));

    // Allocates a temporary Tensor of the specified type and shape. The
    // Tensor must not be used after kernel construction is
    // complete. See comment above.
    
    /*
    Status allocate_temp(DataType type, const TensorShape& shape,
                       Tensor* out_temp);
    */

    Tensor* Aconc = nullptr;
    Tensor* Bconc = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT32, out_shape, Aconc));
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT32, out_shape, Bconc));

    if (out->NumElements() == 0) {
      // If a has shape [0, x] or b has shape [x, 0], the output shape
      // is a 0-element matrix, so there is nothing to do.
      return;
    }

    if (a.NumElements() == 0 || b.NumElements() == 0) {
      // If a has shape [x, 0] and b has shape [0, y], the
      // output shape is [x, y] where x and y are non-zero, so we fill
      // the output with zeros.
      
      /*
      functor::SetZeroFunctor<Device, T> f;
      f(ctx->eigen_device<Device>(), out->flat<T>());
      */
      return;
    }

    // Call the cuda kernel launcher
    //XnorGemmKernelLauncher( /* input.data(), N, output.data() */ );
    /* https://www.tensorflow.org/versions/r1.0/api_docs/cc/class/tensorflow/tensor#dim_size */
    const int32 m = a.dim_size(1 - dim_pair[0].first);
    const int32 k = a.dim_size(dim_pair[0].first);
    const int32 n = b.dim_size(1 - dim_pair[0].second);

    auto a_flat = a.flat<float>().data();
    auto b_flat = b.flat<float>().data();
    auto Aconc_flat = Aconc->flat<int32>().data();
    auto Bconc_flat = Bconc->flat<int32>().data();
    auto c_flat = out->flat<float>().data();

    #if 0
    ConcatenateRowsFunctor<Device, T>()(
    ctx->eigen_device<Device>(),
    a_flat,
    Aconc_flat,
    m);
    #endif
    #if 0
    ConcatenateColsFunctor<Device, T>()(
    ctx->eigen_device<Device>(),
    b_flat,
    Bconc_flat,    
    m);
    #endif
    #if 1
    XnorGemmFunctor<Device, T>()(
    ctx->eigen_device<Device>(),
    Aconc_flat,
    Bconc_flat,
    c_flat,
    m,
    n,
    k);
    #endif

    #if 0 /* For testing base kernel */
    XnorGemmFunctor<Device, T>()(
    ctx->eigen_device<Device>(),
    a.flat<T>().data(),
    b.flat<T>().data(),
    out->flat<float>().data(),
    m,
    n,
    k);
    #endif
  }
  private:
  bool transpose_a_;
  bool transpose_b_;
};

REGISTER_OP("Gemm")
    //.Attr("T: {float, int32} = DT_FLOAT")
      .Input("a: float")
      .Input("b: float")
      .Output("c: float");

/*
    .Doc(R"doc(
Performs XNOR GEMM with matrices A and B resulting in C

output: A Tensor.
  output = A * B
)doc");
*/

#if 1

#define REGISTER_GPU(T)                                         \
    REGISTER_KERNEL_BUILDER(                                    \
        Name("Gemm").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
        XnorGemmOp<GPUDevice, T>);
REGISTER_GPU(float);

#endif

#if 0
REGISTER_KERNEL_BUILDER(
  Name("Gemm")
  .Device(DEVICE_GPU) 
  .TypeConstraint<int32>("T"),
  XnorGemmOp<GPUDevice, int32>);
REGISTER_KERNEL_BUILDER(
  Name("Gemm")
  .Device(DEVICE_GPU) 
  .TypeConstraint<float>("T"),
  BaseGemmOp<GPUDevice, float>);
#endif