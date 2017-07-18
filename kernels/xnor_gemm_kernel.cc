// xnor_gemm_kernel.cc
#define EIGEN_USE_THREADS
#include "xnor_gemm_kernel.h"
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
//void AddOneKernelLauncher(int* in, int N, int* out);

// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
//void XnorGemmKernelLauncher(unsigned int* A, unsigned int* B, float* C, int m, int n, int k);

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
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

    // Call the cuda kernel launcher
    //XnorGemmKernelLauncher( /* input.data(), N, output.data() */ );
    /* https://www.tensorflow.org/versions/r1.0/api_docs/cc/class/tensorflow/tensor#dim_size */
    const int32 m = a.dim_size(1 - dim_pair[0].first);
    const int32 k = a.dim_size(dim_pair[0].first);
    const int32 n = b.dim_size(1 - dim_pair[0].second);

    // Signature (const T* A, const T* B, T* C, int m, int n, int k)
    XnorGemmFunctor<Device, T>()(
    ctx->eigen_device<Device>(),
    a.flat<T>().data(),
    b.flat<T>().data(),
    out->flat<float>().data(),
    m, //static_cast<int>(m),
    n, //static_cast<int>(n),
    k);

    /*
    XnorGemmFunctor<Device, T>()(
    ctx->eigen_device<Device>(),
    static_cast<int>(input_tensor.NumElements()),
    input_tensor.flat<T>().data(),
    output_tensor->flat<T>().data());
    */


    /*
    int block = 64;
    int grid = N * N / (block * 32)  + 1;

    XnorGemmKernelLauncher(a, b, out, m, n, k);
    */
  }
  private:
  bool transpose_a_;
  bool transpose_b_;
};

REGISTER_OP("XnorGemm")
    .Attr("T: {float, int32} = DT_INT32")
      .Input("a: T")
      .Input("b: T")
      .Output("c: float");

/*
    .Doc(R"doc(
Performs XNOR GEMM with matrices A and B resulting in C

output: A Tensor.
  output = A * B
)doc");
*/

//#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(
  Name("XnorGemm")
  .Device(DEVICE_GPU) 
  .TypeConstraint<int32>("T"),
  XnorGemmOp<GPUDevice, int32>);
//#endif
