// concatenate_rows_op.cc
#define EIGEN_USE_THREADS

#include <stdio.h>
//#include "xnor_gemm_kernel.h"
//#include "base_gemm_kernel.h"
#include "concatenate_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
//#include "tensorflow/core/kernels/fill_functor.h"
//#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
//template <typename Device>
template <typename Device, typename T>
class ConcatenateRowsOp : public OpKernel {
 public:
  explicit ConcatenateRowsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  /* Compute method must be thread-safe */
  void Compute(OpKernelContext* ctx) override {
    
    // Get the input tensors
    const Tensor& fA = ctx->input(0);
    const Tensor& size = ctx->input(1);

    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(fA.shape()),
                errors::InvalidArgument("In[0] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(size.shape()),
                errors::InvalidArgument("In[1] is not a scalar"));
    
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
    
    printf("dim_pair[0].first = %d\n", dim_pair[0].first)
    /*
    dim_pair[0].first = 0;
    dim_pair[0].second = 1;
    */
    /*
    OP_REQUIRES(
        ctx, a.dim_size(dim_pair[0].first) == b.dim_size(dim_pair[0].second),
        errors::InvalidArgument(
            "Matrix size-incompatible: In[0]: ", a.shape().DebugString(),
            ", In[1]: ", b.shape().DebugString()));
    */
    int a_dim_remaining = 1 - dim_pair[0].first;
    int b_dim_remaining = 1 - dim_pair[0].second;
    
    TensorShape out_shape({a.dim_size()});
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
    /*
    Tensor* Aconc = nullptr;
    Tensor* Bconc = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT32, out_shape, Aconc));
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT32, out_shape, Bconc));
    */

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

    #if 0
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

REGISTER_OP("ConcatenateRowsOp")
    //.Attr("T: {float, int32} = DT_FLOAT")
      .Input("f_a: float")
      .Input("size: float")
      .Output("a_conc: float")
      .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
      return Status::OK();
    });

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
        Name("ConcatenateRowsOp").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
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