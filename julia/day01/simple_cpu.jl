using Test

N = 2^20;
x = fill(1.0f0,N); # a vector filled with 1.0 (Float32)
y = fill(2.0f0,N);

# 预编译
y .+= x;
println(@test all(y .== 3.0f0));

# 测试时间
start = time();
y .+= x;  # increment each element of y with the corresponding element of x
println(time()-start); # 0.0019638538360595703s

