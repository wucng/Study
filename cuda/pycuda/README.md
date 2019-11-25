- https://documen.tician.de/pycuda/
- https://github.com/inducer/pycuda/tree/master/examples

--- 
[toc]

---

```c
from pycuda.driver import Stream
from pycuda.compiler import DynamicSourceModule

func(a_gpu, block=(4, 4, 1), grid=(1, 1), shared=0,stream=Stream(0))

# 设置使用哪块GPU
cuda.Device(0)
```

# 1.安装
```c
pip3 install pycuda -i https://pypi.doubanio.com/simple
```

# 2.mem_alloc,memcpy_htod,memcpy_dtoh
```python
# 使用一般的开辟内存方式

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy

# a = numpy.random.randn(4,4)
a=numpy.ones([4,4])

a = a.astype(numpy.float32)

a_gpu = cuda.mem_alloc(a.nbytes) # 开辟GPU内存

cuda.memcpy_htod(a_gpu, a) # CPU-->GPU 

# Executing a Kernel
mod = SourceModule("""
  const int N=16;
  __global__ void doublify(float *a)
  {
    int idx = threadIdx.x + threadIdx.y*blockDim.x;
	  if(idx>=N) return;
    a[idx] *= 2;
  }
  """)


func = mod.get_function("doublify")
func(a_gpu,grid=(1,1,1), block=(32,32,1)) # 调用核函数

a_doubled = numpy.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu) # GPU-->CPU
print(a_doubled)
print("\n")
print(a)
```

# 3.pycuda.driver.InOut
```python
# driver.InOut 自动在GPU开辟内存，GPU与CPU都能访问，类似于统一虚拟地址

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy

# a = numpy.random.randn(4,4)
a=numpy.ones([4,4])

a = a.astype(numpy.float32)
"""
a_gpu = cuda.mem_alloc(a.nbytes) # 开辟GPU内存
cuda.memcpy_htod(a_gpu, a) # CPU-->GPU
"""

# Executing a Kernel
mod = SourceModule("""
  const int N=16;
  __global__ void doublify(float *a)
  {
    int idx = threadIdx.x + threadIdx.y*blockDim.x;
	if(idx>=N) return;
    a[idx] *= 2;
  }
  """)

a_gpu=cuda.InOut(a) # a既是输入也是输出
func = mod.get_function("doublify")
func(a_gpu,grid=(1,1,1), block=(32,32,1)) # 调用核函数
"""
a_doubled = numpy.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu) # GPU-->CPU
"""
print(a_gpu)
print("\n")
print(a)
```
# pycuda.driver.In, pycuda.driver.Out
```python
#  pycuda.driver.In, pycuda.driver.Out
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy

# a = numpy.random.randn(4,4)
a=numpy.ones([4,4])

a = a.astype(numpy.float32)

# Executing a Kernel
mod = SourceModule("""
  const int N=16;
  __global__ void doublify(float *a_in,float *a_out)
  {
    int idx = threadIdx.x + threadIdx.y*blockDim.x;
	if(idx>=N) return;
    a_out[idx]=2*a_in[idx];
  }
  """)
a_doubled = numpy.empty_like(a)
a_in=cuda.In(a) # 输入
a_out=cuda.Out(a_doubled) # 输出

func = mod.get_function("doublify")
func(a_in,a_out,grid=(1,1,1), block=(32,32,1)) # 调用核函数

print(a_doubled)
print("\n")
print(a)
```

# 4.pycuda.gpuarray.GPUArray
```python
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy

a_gpu = gpuarray.to_gpu(numpy.random.randn(4,4).astype(numpy.float32))
a_doubled = (2*a_gpu).get() # .get() ,gpu-->cpu
print(a_doubled)
print(a_gpu)
```

# 5.pycuda.driver.to_device, pycuda.driver.from_device
```python
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy

# a = numpy.random.randn(4,4)
a=numpy.ones([4,4])

a = a.astype(numpy.float32)

# Executing a Kernel
mod = SourceModule("""
  const int N=16;
  __global__ void doublify(float *a_in,float *a_out)
  {
    int idx = threadIdx.x + threadIdx.y*blockDim.x;
	if(idx>=N) return;
    a_out[idx]=2*a_in[idx];
  }
  """)
a_doubled = numpy.empty_like(a)
a_in=cuda.to_device(a) # CPU-->GPU
a_out=cuda.to_device(a_doubled) # CPU-->GPU

func = mod.get_function("doublify")
func(a_in,a_out,grid=(1,1,1), block=(32,32,1)) # 调用核函数

a_doubled=cuda.from_device(a_out,a.shape,a.dtype) # GPU-->CPU

print(a_doubled)
print("\n")
print(a)
```
# 6.共享内存
```python
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy

# a = numpy.random.randn(4,4)
a=numpy.ones([4,4])

a = a.astype(numpy.float32)

# Executing a Kernel
mod = SourceModule("""
  const int N=16;
  __global__ void doublify(float *a_in,float *a_out)
  {
    __shared__ float sdatas[N];
    int idx = threadIdx.x + threadIdx.y*blockDim.x; // 对应全局内存id
    int tid=threadIdx.x; // 对应共享内存id
	if(idx>=N) return;
	// 写入共享内存中
	sdatas[tid]=2*a_in[idx];
	__syncthreads();
	// 共享内存写入全局内存
    // a_out[idx]=2*a_in[idx];
    a_out[idx]=sdatas[tid];
  }
  """)
a_doubled = numpy.empty_like(a)
a_in=cuda.to_device(a) # CPU-->GPU
a_out=cuda.to_device(a_doubled) # CPU-->GPU

func = mod.get_function("doublify")
func(a_in,a_out,grid=(1,1,1), block=(32,32,1)) # 调用核函数

a_doubled=cuda.from_device(a_out,a.shape,a.dtype) # GPU-->CPU

print(a_doubled)
print("\n")
print(a)
```


# 7.与numpy比较
```python
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
import numpy

start=time.time()
print("求数组平方和\n")

h=4000
w=4000
N=h*w
a=numpy.ones([h,w])
a = a.astype(numpy.float32)
using_cuda=True

if using_cuda:
    # Executing a Kernel
    mod = SourceModule("""
      const int N=4000*4000;
      __global__ void doublify(float *a_in,float *a_out)
      {
        __shared__ float sdatas[256];
        int idx = threadIdx.x + blockIdx.x*blockDim.x; // 对应全局内存id
        int tid=threadIdx.x; // 对应共享内存id
        if(idx>=N) return;
        // 写入共享内存
        sdatas[tid]=a_in[idx]*a_in[idx];
        __syncthreads();
        // 共享内存写入全局内存
        atomicAdd(&a_out[0],sdatas[tid]);
        
        // atomicAdd(&a_out[0],a_in[idx]*a_in[idx]); // 直接从全局内存中取数据
      }
      """)
    a_doubled = numpy.zeros([1],dtype=numpy.float32)
    a_in=cuda.to_device(a) # CPU-->GPU
    a_out=cuda.to_device(a_doubled) # CPU-->GPU

    func = mod.get_function("doublify")
    func(a_in,a_out,grid=(N//256+1,1,1), block=(256,1,1)) # 调用核函数
    a_doubled=cuda.from_device(a_out,a_doubled.shape,a_doubled.dtype) # GPU-->CPU
else:
    a_doubled=numpy.sum(a*a)

print(a_doubled)

print("cost time:%s"%(time.time()-start))

"""
cuda:  cost time:0.16940021514892578
numpy: cost time:0.08401870727539062
"""

```


# 8.DynamicSourceModule
```python
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
    // const int N=20; // 也可以在外部定义
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

```
# 9.ElementwiseKernel
```python
from __future__ import absolute_import
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import numpy
from pycuda.curandom import rand as curand

a_gpu = curand((50,))
b_gpu = curand((50,))

from pycuda.elementwise import ElementwiseKernel
lin_comb = ElementwiseKernel(
        "float a, float *x, float b, float *y, float *z",
        "z[i] = my_f(a*x[i], b*y[i])",
        "linear_combination",
        preamble="""
        __device__ float my_f(float x, float y)
        { 
          return sin(x*y);
        }
        """)

c_gpu = gpuarray.empty_like(a_gpu)
lin_comb(5, a_gpu, 6, b_gpu, c_gpu)

import numpy.linalg as la
assert la.norm(c_gpu.get() - numpy.sin((5*a_gpu*6*b_gpu).get())) < 1e-5
```


# 10.打印显卡信息
```python
"""
https://github.com/inducer/pycuda/blob/master/examples/dump_properties.py
"""
from __future__ import print_function
from __future__ import absolute_import
import pycuda.driver as drv
from six.moves import range

drv.init()
print("%d device(s) found." % drv.Device.count())

for ordinal in range(drv.Device.count()):
    dev = drv.Device(ordinal)
    print("Device #%d: %s" % (ordinal, dev.name()))
    print("  Compute Capability: %d.%d" % dev.compute_capability())
    print("  Total Memory: %s KB" % (dev.total_memory() // (1024)))
    atts = [(str(att), value)
            for att, value in list(dev.get_attributes().items())]
    atts.sort()

    for att, value in atts:
        print("  %s: %s" % (att, value))
```

# 11.结构体数组
```python
# https://github.com/inducer/pycuda/blob/master/examples/demo_struct.py
# prepared invocations and structures -----------------------------------------
from __future__ import print_function
from __future__ import absolute_import
import pycuda.driver as cuda
import pycuda.autoinit
import numpy
from pycuda.compiler import SourceModule

class DoubleOpStruct:
    mem_size = 8 + numpy.uintp(0).nbytes
    def __init__(self, array, struct_arr_ptr):
        self.data = cuda.to_device(array)
        self.shape, self.dtype = array.shape, array.dtype
        """
        numpy.getbuffer() needed due to lack of new-style buffer interface for
        scalar numpy arrays as of numpy version 1.9.1
        see: https://github.com/inducer/pycuda/pull/60
        """
        cuda.memcpy_htod(int(struct_arr_ptr),
                         numpy.getbuffer(numpy.int32(array.size)))
        cuda.memcpy_htod(int(struct_arr_ptr) + 8,
                         numpy.getbuffer(numpy.uintp(int(self.data))))

    def __str__(self):
        return str(cuda.from_device(self.data, self.shape, self.dtype))

struct_arr = cuda.mem_alloc(2 * DoubleOpStruct.mem_size)
do2_ptr = int(struct_arr) + DoubleOpStruct.mem_size

array1 = DoubleOpStruct(numpy.array([1, 2, 3], dtype=numpy.float32), struct_arr)
array2 = DoubleOpStruct(numpy.array([0, 4], dtype=numpy.float32), do2_ptr)

print("original arrays")
print(array1)
print(array2)

mod = SourceModule("""
    struct DoubleOperation {
        int datalen, __padding; // so 64-bit ptrs can be aligned
        float *ptr;
    };
    __global__ void double_array(DoubleOperation *a)
    {
        a = a + blockIdx.x;
        for (int idx = threadIdx.x; idx < a->datalen; idx += blockDim.x)
        {
            float *a_ptr = a->ptr;
            a_ptr[idx] *= 2;
        }
    }
    """)
func = mod.get_function("double_array")
func(struct_arr, block=(32, 1, 1), grid=(2, 1))

print("doubled arrays")
print(array1)
print(array2)

func(numpy.uintp(do2_ptr), block=(32, 1, 1), grid=(1, 1))
print("doubled second only")
print(array1)
print(array2)

if cuda.get_version() < (4, ):
    func.prepare("P", block=(32, 1, 1))
    func.prepared_call((2, 1), struct_arr)
else:
    func.prepare("P")
    block = (32, 1, 1)
    func.prepared_call((2, 1), block, struct_arr)


print("doubled again")
print(array1)
print(array2)

if cuda.get_version() < (4, ):
    func.prepared_call((1, 1), do2_ptr)
else:
    func.prepared_call((1, 1), block, do2_ptr)


print("doubled second only again")
print(array1)
print(array2)
```
