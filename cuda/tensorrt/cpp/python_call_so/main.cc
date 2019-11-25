#include <iostream>
#include "cSample.h"

int main()
{   
    const int n=5;
    float a[n]={0,1,-2,-5,8};
    int ret=relu(n,a);
    for(auto c:a)
        std::cout<<c<<" ";
    std::cout<<std::endl;
    return 0;
}
