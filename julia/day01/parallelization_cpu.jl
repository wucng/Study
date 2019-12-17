using Test

N = 2^20;
x = fill(1.0f0,N); # a vector filled with 1.0 (Float32)
y = fill(2.0f0,N);

function parallel_add!(y, x)
    Threads.@threads for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

# fill!(y, 2);

parallel_add!(y, x);
println(@test all(y .== 3.0f0));

# 测时间
start = time();
parallel_add!(y, x);
println(time()-start);
