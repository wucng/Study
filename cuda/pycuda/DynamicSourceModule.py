"""
1.多流处理
2.共享内存
"""

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import DynamicSourceModule
from pycuda.compiler import SourceModule
from pycuda.driver import Stream

# 设置使用哪块GPU
cuda.Device(0)

# or DynamicSourceModule --> SourceModule
mod=DynamicSourceModule('''
    #include <cstdio>
    #include <cuda.h>
    
    __device__ void add(float &a)
    {
        a+=10;
    }
    
    // 全局内存
    __global__ void global_add(float *a_inOut,int *shape)
    {
        const int n=shape[0];
        int idx=threadIdx.x+blockDim.x*blockIdx.x;
        // if(idx>=n) return;
        while(idx<n)
        {
           // a_inOut[idx]+=10;
           add(a_inOut[idx]);
           idx+=blockDim.x*gridDim.x;
        }
    }
    
    
    // 共享内存(无法动态传入共享内存大小)
    // const int N=20;
    __global__ void shared_add1(float *a_inOut,int *shape)
    {   
        
        const int N=20; // 必须赋值常数，否则就报错 ,如: const int N=shape[0] 报错
        __shared__ float sdatas[N];
        const int n=shape[0];
        int idx=threadIdx.x+blockDim.x*blockIdx.x;
        int tid =threadIdx.x;
        if(idx>=n) return;
        // global -->shared
        sdatas[tid]=a_inOut[idx]+10;
        __syncthreads();
        
        // shared-->global
        a_inOut[idx]=sdatas[tid];
    }
    
    // 共享内存(方式二，可以动态传入共享内存大小)
    __global__ void shared_add2(float *a_inOut,int *shape)
    {   
        extern __shared__ float sdatas[]; // 声明了共享内存，内存大小需要传递进来
        const int n=shape[0];
        int idx=threadIdx.x+blockDim.x*blockIdx.x;
        int tid =threadIdx.x;
        if(idx>=n) return;
        // global -->shared
        sdatas[tid]=a_inOut[idx]+10;
        __syncthreads();
        
        // shared-->global
        a_inOut[idx]=sdatas[tid];
    }
    '''
)

def main():
    x=np.arange(0,20,dtype=np.float32)
    g_x = cuda.to_device(x)  # cup-->gpu

    """
    # -----------------global memory-------------------------------------------------------
    func=mod.get_function("global_add")
    func(g_x,cuda.to_device(np.asarray(x.shape,dtype=np.int32)),
         grid=((x.size+512-1)//512,1,1),block=(512,1,1),shared=0,stream=Stream(0))
    """
    """
    # -----------------shared memory-------------------------------------------------------
    # --------在gpu内部设置了共享内存的大小，外部可以不用传递,因此可以设置shared=0-------------------------------------
    func = mod.get_function("shared_add1")
    func(g_x, cuda.to_device(np.asarray(x.shape, dtype=np.int32)),
         grid=((x.size + 512 - 1) // 512, 1, 1), block=(512, 1, 1), shared=0, stream=Stream(0))

    # """
    # -----------------shared memory-------------------------------------------------------
    # --------在gpu内部声明了共享内存，具体共享内存大小需通过外部传递,因此设置 shared=x.size-------------------------------------
    func = mod.get_function("global_add")
    func(g_x, cuda.to_device(np.asarray(x.shape, dtype=np.int32)),
         grid=((x.size + 512 - 1) // 512, 1, 1), block=(512, 1, 1), shared=x.size, stream=Stream(0))
    # """

    x=cuda.from_device(g_x,x.shape,x.dtype) # gpu-->cpu
    print(x)


if __name__=="__main__":
    main()
