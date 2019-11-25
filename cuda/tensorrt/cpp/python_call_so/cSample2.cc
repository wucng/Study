#include "cSample2.h" 
#include <iostream>

extern "C"{
    int relu(int n,float* a_inOut)
    {
        for(int i=0;i<n;++i)
        {
            a_inOut[i]=a_inOut[i]<0?0:a_inOut[i];
            // std::cout<<a_inOut[i]<<" ";
        }
        // std::cout<<std::endl;
	return 0;
    }
}
