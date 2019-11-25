"""
ar[np.float32_t, ndim=3, mode='strided'] arr 等价与 float[:,:,:] arr
"""

from numpy cimport ndarray as ar
cimport numpy as np
cimport cython

cdef extern from "cSample.h":
    cdef int relu(int,float*)

"""
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef py_relu(int n,ar[np.float32_t,cast=True] arr): # np.uint8_t
    cdef float *bptr
    bptr=<float *>&arr[0] # 得到指针，<float *>类型转换
    return relu(n,bptr)
"""

# or
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef py_relu(int n,float[:] arr): # np.uint8_t
    cdef float *bptr
    bptr=<float *>&arr[0] # 得到指针，<float *>类型转换
    return relu(n,bptr)

# if __name__=="__main__":
#    print(py_relu(3,4))
