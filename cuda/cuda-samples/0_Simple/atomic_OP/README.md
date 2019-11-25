- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

---
```c
// SMS ?= 60 61 70 75
// nvcc systemWideAtomics.cu -arch compute_75 -code sm_75 -I ../../common/  -std=c++11 -w

atomicAdd // 读全局或共享内存地址(变量)，可以跨`block`
atomicAdd_system // 保证该指令相对于系统中的其他CPU和GPU是原子的 等价于atomicAdd
atomicAdd_block // 该指令仅对于同一线程块中其他线程的原子是原子的 不能跨block
atomicCAS() //(Compare And Swap)

atomicSub() // 减法

atomicExch() //
atomicMin()
atomicMax()
atomicInc() // ((old >= val) ? 0 : (old+1))
atomicDec() // (((old == 0) || (old > val)) ? val : (old-1) )

int atomicCAS(int* address, int compare, int val);
atomicCAS() // (old == compare ? val : old)

atomicAnd() //(old & val)
atomicOr() // (old | val)
atomicXor() // (old ^ val)
```
