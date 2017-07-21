#if GOOGLE_CUDA

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#define BLOCK_SIZE 16

#include <stdio.h>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "concatenate_kernel.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

#define EIGEN_USE_GPU


// 32 single float array ->  32 bits int
__device__ int concatenate(const float* array)
{
    int rvalue=0;
    int sign;
    
    for (int i = 0; i < 32; i++)
    {
        sign = (array[i]>=0);
        rvalue = rvalue | (sign<<i);
    }
    
    return rvalue;
}

template <typename T>
__global__ void concatenate_rows_kernel(const float *a, int *b, const int size)
{ 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<size) b[i] = concatenate(&a[i*32]);
}

template <typename T>
__global__ void concatenate_cols_kernel(const float *a, int *b, const int m, const int n)
{   

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(j<n){
        float * array = new float[32];
        for(int i=0; i<m; i+=32){
            for(int k=0; k<32;k++) array[k] = a[j + n*(i+k)];
            b[j+n*i/32]=concatenate(array); 
        } 
        delete[] array;
    }
}

// 32 bits int -> 32 single float array
// TODO: the array allocation should not be done here
__device__ float* deconcatenate(int x)
{
    float * array = new float[32];
    
    for (int i = 0; i < 32; i++)    
    {   
        array[i] = (x & ( 1 << i )) >> i;
    }
    
    return array;
}

__global__ void deconcatenate_rows_kernel(int *a, float *b, int size)
{ 
    float * array;
    
    for(int i=0; i<size; i+=32)
    {
        array = deconcatenate(a[i/32]);
        for (int k=0;k<32;k++) b[i+k] = array[k];
        delete[] array;
    }
}

template <typename T>
struct ConcatenateRowsFunctor<GPUDevice, T> {
    void operator()(const GPUDevice& d, const float* fA, int* Aconc, const int N) {
#ifdef DEBUG        
        printf("\n\nConcatenateRowsFunctor\n\n");
#endif        
        int block = BLOCK_SIZE * 4, grid = N * N / (block * 32)  + 1;
        concatenate_rows_kernel<T>
            <<<grid, block, 0, d.stream()>>>(fA, Aconc, N * N / 32);
    }
};

template <typename T>
struct ConcatenateColsFunctor<GPUDevice, T> {
    void operator()(const GPUDevice& d, const float* fB, int* Bconc, const int N) {
#ifdef DEBUG        
        printf("\n\nConcatenateColsFunctor\n\n");
#endif        
        int block = BLOCK_SIZE * 4;
        int grid = N / block + 1;
        concatenate_cols_kernel<T>
            <<<grid, block, 0, d.stream()>>>(fB, Bconc, N, N);
    }
};

// Instantiate functors for the types of OpKernels registered.
typedef Eigen::GpuDevice GPUDevice;
template struct ConcatenateRowsFunctor<GPUDevice, float>;
template struct ConcatenateColsFunctor<GPUDevice, float>;

#endif  // GOOGLE_CUDA