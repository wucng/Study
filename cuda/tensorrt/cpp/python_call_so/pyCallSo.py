"""
ctypes的运用 : https://blog.csdn.net/MCANDML/article/details/80426914
python调用.so : https://www.cnblogs.com/fariver/p/6573112.html
"""

import os
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

CLIP_PLUGIN_LIBRARY = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'libcSample.so'
)
# so=ctypes.CDLL(CLIP_PLUGIN_LIBRARY)
so = ctypes.cdll.LoadLibrary(CLIP_PLUGIN_LIBRARY)

# '''
pyarray = [-1, 2, -3, 4, 5]
pyarray = np.asarray(pyarray,dtype=np.float32)
size=pyarray.size
fun=so.relu

# ndpointer(ctypes.c_float) 指针类型
fun.argtypes=[ctypes.c_int, ndpointer(ctypes.c_float)]
fun.restype = None # 返回值类型
fun(size,pyarray) # 调用函数
# print(np.array(pyarray)) # 把c array转换为np.array
print(pyarray)

# [0. 2. 0. 4. 5.]