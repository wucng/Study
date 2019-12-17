using CuArrays
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

@time y_d .+= x_d; # 0.000074 s
@test all(Array(y_d) .== 3.0f0)

# or
function add_broadcast!(y, x)
    CuArrays.@sync y .+= x # 这种会同步，导致时间变慢
    return
end
@time add_broadcast!(y_d, x_d); # 0.000291 s

