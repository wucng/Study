import numpy as np
import pycuda.autoinit
from pycuda.autoinit import context
import pycuda.driver as cuda
# from pycuda.compiler import DynamicSourceModule
from pycuda.compiler import SourceModule
from pycuda.driver import Stream
import time


def gpu_min(data: np.array):
    mod = SourceModule("""
        __device__ void swap(float &a,float &b)
        {
            float tmp = a;
            a = b;
            b = tmp;
        }

        // 归约的方式寻找最小值
        __global__ void gmin(float *data,int *shape)
        {
            // extern __shared__ float sdatas[]; // 声明共享变量
            // 每次对折一半
            int len_data = shape[0];
            int tx = threadIdx.x;
            int bx = blockIdx.x;
            int idx = tx + bx * blockDim.x;
            // int lidx = tx + bx * blockDim.x;
            // int ridx = tx + len_data/2 + bx * blockDim.x;
            
            // 每次让前半部分与后半部分交换 缩减一半
            for(int i = len_data/2;i>1;i/=2) // i>>=1
            {
                if (tx<i)
                {
                    if (data[idx]>data[idx+i])
                        swap(data[idx],data[idx+i]);
                }
            }
        }
        
        // 使用共享内存
        __global__ void smin(float *data,int *shape)
        {
            extern __shared__ float sdatas[]; // 声明共享变量
            // 每次对折一半
            // int len_data = shape[0];
            int tx = threadIdx.x;
            int bx = blockIdx.x;
            int idx = tx + bx * blockDim.x;
            
            // 写入共享内存(写入时先做一次对折，元素减半)
            if (tx<blockDim.x/2) //  && idx < len_data/2
            {            
                sdatas[tx] = data[idx]>data[idx+blockDim.x/2]?data[idx+blockDim.x/2]:data[idx];  
            }
            __syncthreads();    
            
            // 每次让前半部分与后半部分交换 缩减一半
            for(int i = blockDim.x/4;i>1;i>>=1) // i/=2
            {
                if (tx<i)
                {
                    if (sdatas[tx]>sdatas[tx+i])
                        swap(sdatas[tx],sdatas[tx+i]);
                }
            }

            // 每个block都得到一个局部最小值，再次归约得到最终的最小值
            if(tx==0)
                data[idx] = sdatas[tx];
        }
    """)

    g_data = cuda.to_device(data)
    data_shape = data.shape
    g_shape = cuda.to_device(np.asarray(data_shape,np.int32))

    """
    func = mod.get_function("gmin")
    block = (2**7,1,1) # 数据长度不限制
    grid = ((len(data)+block[0]-1)//block[0],1,1)
    func(g_data, g_shape, grid=grid, block=block, shared=0, stream=Stream(0))
    """
    func = mod.get_function("smin")
    block = (2**7, 1, 1) # 注：如果数据长度不足2**7 必须填充成2**7
    grid = ((len(data) + block[0] - 1) // block[0], 1, 1)
    func(g_data, g_shape, grid=grid, block=block, shared=block[0], stream=Stream(0))
    # """

    data = cuda.from_device_like(g_data, data)

    return data[0] # 最小值


def main():
    # data = np.asarray([23, 35, 12, 78, 45, 67, 33, 49]).astype(np.float32)
    data = np.ones([2**7],np.float32)*5
    data[0]=2

    start = time.time()
    data1 = np.min(data)
    print(data1)
    print(time.time() - start)

    start = time.time()
    data2 = gpu_min(data)
    print(data2)
    print(time.time() - start)


if __name__ == '__main__':
    main()
