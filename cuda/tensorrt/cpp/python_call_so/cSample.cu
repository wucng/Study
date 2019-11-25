#include "cSample.h" 
#include <cuda.h>

template <typename T>
__global__ void gpu_relu(int n,T* d_arr)
{
    int idx=threadIdx.x+blockDim.x*blockIdx.x;
    while(idx<n)
    {
        d_arr[idx]=d_arr[idx]<0?0:d_arr[idx];
        idx += gridDim.x * blockDim.x;
    }
}

int relu(int n,float* a_inOut)
{
    // cpu-->gpu
    float *d_inOut=NULL;
    cudaMalloc((void**)&d_inOut,n*sizeof(float));
    cudaMemcpy(d_inOut, a_inOut, n*sizeof(float), cudaMemcpyHostToDevice);

    const int blockSize = 512;
    const int gridSize = (n + blockSize - 1) / blockSize;
    gpu_relu<float><<<gridSize,blockSize>>>(n,d_inOut);

    // GPU-->CPU
    cudaMemcpy(a_inOut,d_inOut, n*sizeof(float), cudaMemcpyDeviceToHost);

    // free
    cudaFree(d_inOut);
    
    return 0;
}
