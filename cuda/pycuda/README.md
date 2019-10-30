- https://documen.tician.de/pycuda/

--- 
[toc]

# 安装
```c
pip3 install pycuda -i https://pypi.doubanio.com/simple
```

# mem_alloc,memcpy_htod,memcpy_dtoh
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

# pycuda.driver.InOut
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
# pycuda.driver.In, pycuda.driver.Ou
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

# pycuda.gpuarray.GPUArray
```python
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy

a_gpu = gpuarray.to_gpu(numpy.random.randn(4,4).astype(numpy.float32))
a_doubled = (2*a_gpu).get()
print(a_doubled)
print(a_gpu)
```

# pycuda.driver.to_device, pycuda.driver.from_device
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
# 共享内存
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


# 与numpy比较
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