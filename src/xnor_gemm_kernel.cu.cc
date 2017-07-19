#if GOOGLE_CUDA

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#define BLOCK_SIZE 16

#include <stdio.h>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "xnor_gemm_kernel.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

#define EIGEN_USE_GPU

template <typename T>
// CUDA tutorial: http://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
__global__ void xnor_gemm(const int* A, const int* B, float* C, const int m, const int n, const int k) {
    
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ int As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    int Cvalue = 0;
    
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {
    
        // Get sub-matrix Asub of A
        const int* Asub = &A[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];
        
        // Get sub-matrix Bsub of B
        const int* Bsub = &B[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];
        
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = Asub[row*n+col];
        Bs[row][col] = Bsub[row*k+col];
    
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        
        // Multiply Asub and Bsub together
        // THIS IS THE MOST INTERESTING PART
        for (int j = 0; j < BLOCK_SIZE; ++j) Cvalue += __popc(As[row][j]^Bs[j][col]);
        
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    
    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m) Csub[row*k+col] = -(2*(float)Cvalue-32*n);
}

/*
void XnorGemmKernelLauncher(unsigned int* A, unsigned int* B, float* C, int m, int n, int k, int grid, int block) {
    //xnor_gemm<<<gridDim, blockDim>>>(A, B, C, N, N / 32, N);
    //int block_size = 16;
    //int grid = (k / block_size + 1, m / block_size + 1);
    xnor_gemm<<<grid, block>>>(A, B, C, m, n, k);
}*/

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct XnorGemmFunctor<GPUDevice, T> {
//void operator()(const GPUDevice& d, int size, const T* in, T* out) {
//void operator()(const GPUDevice& d, const T* A, const T* B, float* C, const T* m, const T* n, const T* k) {
  void operator()(const GPUDevice& d, const int* A, const int* B, float* C, const int m, const int n, const int k) {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    printf("\n\nInt32 input -- using XnorGemmFunctor\n\n");
    int block_count = BLOCK_SIZE;
    int thread_per_block = 512;
    xnor_gemm<T>
        <<<block_count, thread_per_block, 0, d.stream()>>>(A, B, C, m, n, k);
  }
};

// Instantiate functors for the types of OpKernels registered.
typedef Eigen::GpuDevice GPUDevice;
template struct XnorGemmFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA