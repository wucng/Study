#include <thrust/host_vector.h> // 对应 host上的vector 类似 STL::vector
#include <thrust/device_vector.h> // 对应GPU上的vector
#include <iostream>
#include "cuda.h"
#include "common/book.h"
#define mycout cout<<"["<<__FILE__<<":"<<__LINE__<<"] "
#define get_tid() (blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x)  // 2D grid,1D block

using namespace std;
typedef float FLOAT;

int main()
{
    mycout<<"结合thrust库实现矩阵与向量相乘\n"
    << "C=A*b"<<endl;

    const int height=3,width=5;
    thrust::device_vector<FLOAT> A(height*width);
    // 赋值
    for(int i=0;i<A.size();++i)
        A[i]=(FLOAT)1;

    // print contents of A
    for(int i = 0; i <A.size(); i++)
   {
      cout << A[i] <<" ";
      if((i+1)%width==0) cout<<endl;
   }
   cout<<endl;

    thrust::host_vector<FLOAT> b1(width);
    for(int i=0;i<b1.size();++i)
        b1[i]=(FLOAT)1;

    // Copy host_vector b1 to device_vector b
    thrust::device_vector<FLOAT> b=b1;
    thrust::device_vector<FLOAT> C1(height*width);
    thrust::device_vector<FLOAT> C(height);

    for(int i=0;i<height;++i)
    {
        for(int j=0;j<width;++j)
        {
            C1[j+i*width]=A[j+i*width]*b[j];
        }
		// 再将C1的每一行累加
        C[i]=thrust::reduce(C1.begin()+i*width, C1.begin()+(i+1)*width, (FLOAT) 0, thrust::plus<FLOAT>());
    }


    // print
    for(int i=0;i<C.size();++i)
        cout<<C[i]<<" ";

    return 0;
}
