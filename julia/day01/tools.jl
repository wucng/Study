using CUDAnative
using CuArrays
using Test
using CUDAdrv
# println(CUDAdrv.name(CuDevice(0)));
dev = CuDevice(0);
ctx = CuContext(dev);

N = 2^20;
x_d = CuArrays.fill(1.0f0, N);
y_d = CuArrays.fill(2.0f0, N);
