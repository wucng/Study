#include <iostream>

using namespace std;
typedef float FLOAT;
int main()
{
    int rows=5,cols=3;
    FLOAT **a=nullptr;
    // 分配内存
    a=(FLOAT**)malloc(rows*sizeof(FLOAT*));
    for(int i=0;i<rows;++i)
    {
        a[i]=(FLOAT *)malloc(cols*sizeof(FLOAT));
    }

    // 赋值
    for(int i=0;i<rows;++i)
    {
        for(int j=0;j<cols;++j)
        {
            a[i][j]=j+i*cols;
        }
    }

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
            free(a[i]);
    }
    if(a!=NULL)
        free(a);

    return 0;
}
