using Test
using Distributed
addprocs(4); # 增加4个子进程

N = 2^20;
x = fill(1.0f0,N); # a vector filled with 1.0 (Float32)
y = fill(2.0f0,N);

@everywhere function sequential_add(y, x)
    @simd for i in eachindex(y, x) # @elapsed @simd for i in eachindex(y, x)
        @inbounds y[i] += x[i] # y[i] += x[i]
    end
    return y
end

# fill!(y, 2);

y = fetch(@spawn sequential_add(y, x));
println(@test all(y .== 3.0f0));

# 测时间
start = time();
y = fetch(@spawn sequential_add(y, x));
println(time()-start); # 0.5807
