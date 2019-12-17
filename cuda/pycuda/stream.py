import numpy as np
import pycuda.autoinit
from pycuda.autoinit import context
import pycuda.driver as cuda
# from pycuda.compiler import DynamicSourceModule
from pycuda.compiler import SourceModule
from pycuda.driver import Stream

def multiStream():
    mod = SourceModule("""
        __global__ void add(float *data)
        {
            int idx = threadIdx.x;
            data[idx] += 10.0f;
        }
    """)

    func = mod.get_function("add")

    # a = np.linspace(0,21,20,dtype=np.float32)
    a = np.arange(0,20,1,dtype = np.float32)
    output = np.zeros_like(a)

    g_a1 = cuda.mem_alloc(a.nbytes//2)
    g_a2 = cuda.mem_alloc(a.nbytes//2)

    cuda.memcpy_htod_async(g_a1,a[:10],Stream(0))
    cuda.memcpy_htod_async(g_a2,a[10:],Stream(1))
    context.synchronize()

    func(g_a1,grid=(1,1,1),block=(10,1,1),shared=0,stream=Stream(0))
    func(g_a2,grid=(1,1,1),block=(10,1,1),shared=0,stream=Stream(1))

    cuda.memcpy_dtoh_async(output[:10],g_a1,Stream(0))
    cuda.memcpy_dtoh_async(output[10:],g_a2,Stream(1))
    context.synchronize()

    return output

def multiStream2():
    mod = SourceModule("""
        __global__ void add(float *data)
        {
            int idx = threadIdx.x;
            data[idx] += 10.0f;
        }
    """)

    func = mod.get_function("add")

    # a = np.linspace(0,21,20,dtype=np.float32)
    a = np.arange(0,20,1,dtype = np.float32)
    output = np.zeros_like(a)

    for i in range(2):
        g_a = cuda.mem_alloc(a.nbytes//2)
        cuda.memcpy_htod_async(g_a, a[i*10:(i+1)*10], Stream(i))
        context.synchronize()
        func(g_a, grid=(1, 1, 1), block=(10, 1, 1), shared=0, stream=Stream(i))
        cuda.memcpy_dtoh_async(output[i*10:(i+1)*10], g_a, Stream(i))
        context.synchronize()

    return output


def main():
    output= multiStream2()
    print(output)

if __name__=="__main__":
    main()

