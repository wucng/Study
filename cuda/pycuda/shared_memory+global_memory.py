"""
1.共享内存
共享内存声明: extern __shared__ float sdatas[]; # 只能声明一维
共享内存声明与定义：
__shared__ float sdatas[N]; # 不能初始化
__shared__ float sdatas[N][N];

2.全局内存
- 统一虚拟管理内存，CPU与GPU都可以访问(推荐)
a=cuda.managed_empty(32,np.float32,mem_flags=cuda.mem_attach_flags.GLOBAL)
a[:] = np.ones((32,),np.float32)

- 使用mem_alloc,memcpy_htod,memcpy_dtoh

- pycuda.driver.InOut,pycuda.driver.In,pycuda.driver.Out

- gpuarray.GPUArray（推荐）

- pycuda.driver.to_device，cuda.from_device（cuda.from_device_like） （推荐）
"""


import numpy as np
import pycuda.autoinit
from pycuda.autoinit import context
import pycuda.driver as cuda
# from pycuda.compiler import DynamicSourceModule
from pycuda.compiler import SourceModule
from pycuda.driver import Stream
import pycuda.gpuarray as gpuarray

def sharedMemory():
    mod=SourceModule(
    """
        #define N 32
        __global__ void add(float *a_inOut)
        {
            __shared__ float sdatas[N];
            // __shared__ float sdatas[N][N]; // 这种方式还可以定义二维
            int tid = threadIdx.x;
            int idx = tid + blockIdx.x*blockDim.x;
            // 全局内存-->共享内存
            sdatas[tid] = a_inOut[idx]+10;
            __syncthreads(); // 同步
            
            // 共享内存--->全局内存
            a_inOut[idx] = sdatas[tid];
        }
        
        __global__ void add2(float *a_inOut)
        {
            extern __shared__ float sdatas[];//声明共享内存
            int tid = threadIdx.x;
            int idx = tid + blockIdx.x*blockDim.x;
            // 全局内存-->共享内存
            sdatas[tid] = a_inOut[idx]+10;
            __syncthreads(); // 同步
            
            // 共享内存--->全局内存
            a_inOut[idx] = sdatas[tid];
        }
    """
    )

    """
    # 使用统一虚拟管理内存，CPU与GPU都可以访问
    a=cuda.managed_empty(32,np.float32,mem_flags=cuda.mem_attach_flags.GLOBAL)
    a[:] = np.ones((32,),np.float32)

    func=mod.get_function("add")
    func(a,grid=(1,1,1),block=(32,1,1),shared=0,stream=Stream(0))

    # func = mod.get_function("add2")
    # func(a, grid=(1, 1, 1), block=(32, 1, 1), shared=32, stream=Stream(0)) # shared=32传入共享内存大小，这种方式好像只能声明一维共享内存
    context.synchronize() # Wait for kernel completion before host access
    """

    """
    # 使用mem_alloc,
    a = np.ones((32,),np.float32)
    a_gpu = cuda.mem_alloc(a.nbytes)
    cuda.memcpy_htod(a_gpu, a)  # CPU-->GPU
    func = mod.get_function("add")
    func(a_gpu, grid=(1, 1, 1), block=(32, 1, 1), shared=0, stream=Stream(0))
    cuda.memcpy_dtoh(a, a_gpu)  # GPU--> CPU
    a_gpu.free()
    """

    """
    # 使用pycuda.driver.InOut,pycuda.driver.In,pycuda.driver.Out
    a = np.ones((32,), np.float32)
    a_gpu = cuda.InOut(a)  # a既是输入也是输出
    func = mod.get_function("add")
    func(a_gpu, grid=(1, 1, 1), block=(32, 1, 1), shared=0, stream=Stream(0))
    a = a_gpu.array
    a_gpu.free()
    """

    """
    # gpuarray.GPUArray
    a = np.ones((32,), np.float32)
    a_gpu = gpuarray.to_gpu(a)
    func = mod.get_function("add")
    func(a_gpu, grid=(1, 1, 1), block=(32, 1, 1), shared=0, stream=Stream(0))
    a = a_gpu.get()
    a_gpu.free()
    """

    # """
    # pycuda.driver.to_device
    a = np.ones((32,), np.float32)
    a_gpu = cuda.to_device(a)
    func = mod.get_function("add")
    func(a_gpu, grid=(1, 1, 1), block=(32, 1, 1), shared=0, stream=Stream(0))
    a = cuda.from_device(a_gpu,a.shape,a.dtype)
    # or
    # a = cuda.from_device_like(a_gpu,a)
    # """
    a_gpu.free()
    return a

def main():
    a= sharedMemory()
    print(a)


if __name__=="__main__":
    main()
