#include "cSample.h" 

int relu(int n,float* a_inOut)
{
    for(int i=0;i<n;++i)
    {
        a_inOut[i]=a_inOut[i]<0?0:a_inOut[i];
    }
}
