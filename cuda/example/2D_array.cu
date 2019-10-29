#include <iostream>
#include <cuda.h>

#define mycout cout<<"["<<__FILE__<<":"<<__LINE__<<"] "
#define CHECK(res) if(res!=cudaSuccess){exit(-1);}

using namespace std;
typedef float FLOAT;

__global__ void vec_add(FLOAT **a,const int rows,const int cols)
{
    int x=threadIdx.x;
    int y=threadIdx.y;
    if(x>=cols || y>=rows) return;

    a[y][x]+=2;
}

int main()
{
    mycout<<"虚拟统一内存使用(CPU与GPU都能访问)\n"<<
    "使用二维数组(二维数组其实可以展开为一维数组处理)"<<endl;

    int rows=5,cols=3;
    FLOAT **a=nullptr;
    // 分配内存
    // a=(FLOAT**)malloc(rows*sizeof(FLOAT*));
    CHECK(cudaMallocManaged((void**)&a,rows*sizeof(FLOAT*)));

    for(int i=0;i<rows;++i)
    {
        // a[i]=(FLOAT *)malloc(cols*sizeof(FLOAT));
        CHECK(cudaMallocManaged((void**)&a[i],cols*sizeof(FLOAT)));
    }

    // 赋值
    for(int i=0;i<rows;++i)
    {
        for(int j=0;j<cols;++j)
        {
            a[i][j]=j+i*cols;
        }
    }

    // 启动核函数
    dim3 threads(32,32);
    vec_add<<<1,threads>>>(a,rows,cols);

    cudaDeviceSynchronize(); //等待GPU执行完成， 有多种方式

    // 打印
    for(int i=0;i<rows;++i)
    {
        for(int j=0;j<cols;++j)
        {
            cout<<a[i][j]<<" ";
        }
        cout<<endl;
    }

    // free
    for(int i=0;i<rows;++i)
    {
        if(a[i]!=NULL)
            // free(a[i]);
            CHECK(cudaFree(a[i]));
    }
    if(a!=NULL)
        // free(a);
        CHECK(cudaFree(a));

    return 0;
}
