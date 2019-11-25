// nvcc test.cu  -I ../../common/  -std=c++11

#include "matrixMul.cuh"
// typedef float FLOAT;
#define BLOCK_SIZE 32

int main00(int argc,char *argv[])
{
    // 计算时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Task<FLOAT> task;
    Data2d<FLOAT> task;
    unsigned int m=0,n=0,k=0,id=0;
    m=n=k=BLOCK_SIZE;
    task.allocate(m,n,k,id);
    dim3 block(BLOCK_SIZE,BLOCK_SIZE,1);
    dim3 grid(1,1,1);

    // Time copies and kernel
    cudaEventRecord(start,0);
    // checkError(cudaDeviceSynchronize()); // 同步,CPU等待GPU全部完成
    // MatrixMulCUDA0<BLOCK_SIZE><<<grid,block>>>(task.A,task.B,task.C,m,n,k);
    MatrixMulCUDA1<BLOCK_SIZE><<<grid,block>>>(task.d_a,task.d_b,task.d_c,m,n,k);

    // checkCudaErrors(cudaGetLastError());// launch vectorAdd kernel
    // checkError(cudaDeviceSynchronize()); // 同步,CPU等待GPU全部完成
    task.pprint();

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start, stop);

    mycout<<kernel_time<<" (ms)"<<endl;

    return 0;
}

int main01()
{
    // 计算时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Task<FLOAT> task;
    Data2d<FLOAT> task;
    // unsigned int m=0,n=0,k=0,id=0;
    // m=n=k=BLOCK_SIZE;
    int block_size=32;
    // dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
    // dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);
    dim3 dimsA(block_size,block_size, 1);
    dim3 dimsB(block_size*16, block_size, 1);

    cout<<"A:"<<dimsA.y<<" x "<<dimsA.x<<endl;
    cout<<"B:"<<dimsB.y<<" x "<<dimsB.x<<endl;

    // exit(0);
    task.allocate(dimsA.y,dimsA.x,dimsB.x,0);
    // Setup execution parameters
    dim3 threads(256,1,1);
    // dim3 grid((dimsA.x+256-1)/256,1,1);
    dim3 grid((dimsA.x*dimsA.y+256-1)/256,1,1);

    // Time copies and kernel
    cudaEventRecord(start,0);
    // gpu_matrix_mul1<FLOAT><<<grid,threads>>>(task.d_a,task.d_b,task.d_c,task.m, task.n,task.k);//245.631 (ms)
    gpu_matrix_mul<FLOAT><<<grid,threads>>>(task.d_a,task.d_b,task.d_c,task.m, task.n,task.k);//36.5556 (ms)

    // checkCudaErrors(cudaGetLastError());// launch vectorAdd kernel
    checkError(cudaDeviceSynchronize()); // 同步,CPU等待GPU全部完成
    // task.pprint();

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start, stop);

    mycout<<kernel_time<<" (ms)"<<endl;

    return 0;
}


int main02()
{
    // 计算时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Task<FLOAT> task;
    Data2d<FLOAT> task;
    // unsigned int m=0,n=0,k=0,id=0;
    unsigned int id=0;
    // m=n=k=BLOCK_SIZE;
    int block_size=32;
    dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
    dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);
    // dim3 dimsA(block_size,block_size, 1);
    // dim3 dimsB(block_size*16, block_size, 1);

    cout<<"A:"<<dimsA.y<<" x "<<dimsA.x<<endl;
    cout<<"B:"<<dimsB.y<<" x "<<dimsB.x<<endl;

    // exit(0);
    task.allocate(dimsA.y,dimsA.x,dimsB.x,id);
    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    // Time copies and kernel
    cudaEventRecord(start,0);
    MatrixMulCUDA<32><<<grid,threads>>>(task.d_a,task.d_b,task.d_c,dimsA.y, dimsB.x);//0.047424 (ms)
    // MatrixMulCUDA<16><<<grid,threads>>>(task.d_a,task.d_b,task.d_c,dimsA.y, dimsB.x);

    // checkCudaErrors(cudaGetLastError());// launch vectorAdd kernel
    checkError(cudaDeviceSynchronize()); // 同步,CPU等待GPU全部完成
    // task.pprint();

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start, stop);

    mycout<<kernel_time<<" (ms)"<<endl;

    return 0;
}

int main03()
{
    // 计算时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Task<FLOAT> task;
    Data2d<FLOAT> task;
    // unsigned int m=0,n=0,k=0,id=0;
    // m=n=k=BLOCK_SIZE;
    int block_size=32;
    dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
    dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);
    // dim3 dimsA(block_size,block_size, 1);
    // dim3 dimsB(block_size*16, block_size, 1);

    cout<<"A:"<<dimsA.y<<" x "<<dimsA.x<<endl;
    cout<<"B:"<<dimsB.y<<" x "<<dimsB.x<<endl;

    // exit(0);
    task.allocate(dimsA.y,dimsA.x,dimsB.x,1);
    // task.pprint();
    #if 1
    {
        // Setup execution parameters
        dim3 threads(block_size,block_size,1);
        dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

        // Time copies and kernel
        cudaEventRecord(start,0);
        // 每次只执行1个block
        // MatrixMulSelf<32><<<grid,threads>>>(task.d_a,task.d_b,task.d_c,task.m, task.n,task.k); // 0.294784 (ms)
        // 所有block同时执行
        MatrixMulSelfV2<32><<<grid,threads>>>(task.d_a,task.d_b,task.d_c,task.m, task.n,task.k);  // 0.045056 (ms)
    }
    #else
    {
        // Setup execution parameters
        dim3 threads(block_size/2,block_size/2,1); // 16 x 16
        dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

        // Time copies and kernel
        cudaEventRecord(start,0);
        // 每次只执行1个block
        // MatrixMulSelf<16><<<grid,threads>>>(task.d_a,task.d_b,task.d_c,task.m, task.n,task.k); // 0.294784 (ms)
        // 所有block同时执行
        MatrixMulSelfV2<16><<<grid,threads>>>(task.d_a,task.d_b,task.d_c,task.m, task.n,task.k);  // 0.045056 (ms)
    }
    #endif

    // checkCudaErrors(cudaGetLastError());// launch vectorAdd kernel
    // checkError(cudaDeviceSynchronize()); // 同步,CPU等待GPU全部完成
    task.pprint();

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start, stop);

    mycout<<kernel_time<<" (ms)"<<endl;

    return 0;
}


int main()
{   
    // 矩阵乘以向量
    // 计算时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Task<FLOAT> task;
    Data2d<FLOAT> task;
    int block_size=32;
    // dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
    // dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);
    dim3 dimsA(block_size,block_size*64, 1);
    dim3 dimsB(1, block_size, 1);

    cout<<"A:"<<dimsA.y<<" x "<<dimsA.x<<endl;
    cout<<"B:"<<dimsB.y<<" x "<<dimsB.x<<endl;

    task.allocate(dimsA.y,dimsA.x,dimsB.x,1);

    #if 0
    {
        // Setup execution parameters
        dim3 threads(dimsB.y,1,1);
        dim3 grid(dimsA.y*dimsA.x / threads.x,1,1);

        // Time copies and kernel
        cudaEventRecord(start,0);
        matrix_mul_vector<FLOAT><<<grid,threads>>>(task.d_a,task.d_b,task.d_c,task.m, task.n);// 5.9705 (ms)
        // matrix_mul_vector_sm<32><<<grid,threads>>>(task.d_a,task.d_b,task.d_c,task.m, task.n); // 1.04832 (ms)
    }
    #else
    {
        // Setup execution parameters
        dim3 threads(block_size,block_size,1);
        // dim3 grid(dimsA.y*dimsA.x / threads.x,1,1);
        dim3 grid(1, dimsA.y / threads.y);

        // Time copies and kernel
        cudaEventRecord(start,0);
        matrix_mul_vector_block<32><<<grid,threads>>>(task.d_a,task.d_b,task.d_c,task.m, task.n); // 0.586752 (ms)    
    }
    #endif
    // checkCudaErrors(cudaGetLastError());// launch vectorAdd kernel
    // checkError(cudaDeviceSynchronize()); // 同步,CPU等待GPU全部完成
    // task.pprint();

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start, stop);

    mycout<<kernel_time<<" (ms)"<<endl;

    return 0;
}