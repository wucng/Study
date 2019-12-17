using CuArrays
using CUDAnative
using Test


# cpu
N = 2^20;
x = fill(1.0f0,N); # a vector filled with 1.0 (Float32)
y = fill(2.0f0,N);

# cpu -->GPU
x_d = CuArrays.CuArray(x);
y_d = CuArrays.CuArray(y);

# or 直接创建GPU变量
# x_d = CuArrays.fill(1.0f0, N);
# y_d = CuArrays.fill(2.0f0, N);

# GPU -->CPU  # Array(x_d)
# @test all(Array(x_d) .==1.0f0)

# ------------------单线程---------------------------

function gpu_add1!(y, x)
    for i = 1:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

@time @cuda gpu_add1!(y_d, x_d) # 0.000074 s 这种其实是单线程
# @time @cuda threads=(1,1,1) gpu_add1!(y_d, x_d);
@test all(Array(y_d) .== 3.0f0)


# ----------------自定义多线程---------------------------
function gpu_add2!(y, x)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end
# 等价于
function gpu_add2!(y, x)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    while index <= length(y)
	y[index] += x[index];
        index += stride;
    end
    return nothing
end


fill!(y_d, 2);
@time @cuda threads=(256,1,1) gpu_add2!(y_d, x_d); # 0.000078 s
@test all(Array(y_d) .== 3.0f0)

# ----------------自定义多线程2---------------------------
function gpu_add3!(y, x)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x # julia索引都是从1开始
    stride = blockDim().x * gridDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

# 等价于
function gpu_add3!(y, x)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x # julia索引都是从1开始
    stride = blockDim().x * gridDim().x
    while index <= length(y)
	y[index] += x[index];
        index += stride;
    end
    return nothing
end

numblocks = ceil(Int, N/256);

fill!(y_d, 2);
@time @cuda threads=256 blocks=numblocks gpu_add3!(y_d, x_d) # 0.000077 s
@test all(Array(y_d) .== 3.0f0)


