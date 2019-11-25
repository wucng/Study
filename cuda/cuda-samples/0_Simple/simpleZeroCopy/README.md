```c
// 数据零内存拷贝

// 对于已经预先分配的系统内存区域，可以使用cudaHostRegister()快速固定内存，而无需分配单独的缓冲区并将数据复制到其中。
checkCudaErrors(cudaHostRegister(a, bytes, cudaHostRegisterMapped)); //cpu内存

// 使用Runtime API中的cudaHostAlloc（）函数分配固定的内存。
flags = cudaHostAllocMapped;
checkCudaErrors(cudaHostAlloc((void **)&a, bytes, flags)); // or cudaMallocHost

checkCudaErrors(cudaHostGetDevicePointer((void **)&d_a, (void *)a, 0)); // 内存拷贝 host-->device

// free
checkCudaErrors(cudaHostUnregister(a));
checkCudaErrors(cudaFreeHost(a));

```
