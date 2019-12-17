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

# shared memory
function gpu_add4!(y, x)
    # 声明shared memory
    shared = @cuStaticSharedMem(Float32,256);
    # shared = @cuDynamicSharedMem(Float32,1);
    tid = threadIdx().x;
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x; # julia索引都是从1开始
    stride = blockDim().x * gridDim().x;
    
    index > length(y) && return
    # 全局内存-->shared memory
    shared[tid] = y[index]+x[index];
    # sync_threads(); 同步
    
    # shared memory --> 全局内存
    y[index] = shared[tid];
    return nothing
end

# fill!(y_d,2);

numblocks = ceil(Int,N/256);
@time @cuda threads=(256,1,1) blocks=(numblocks,1,1)  gpu_add4!(y_d, x_d)
@test all(Array(y_d) .== 3.0f0)

destroy!(ctx);
