#ifndef CSAMPLE_H
#define CSAMPLE_H
// #include <iostream>
#include <stdio.h>

// template<typename T> int relu(int n,T* a_inOut);
//extern int relu(int n,float* a_inOut);

extern "C"{
    //int relu(int n,void* a_inOut);
    int relu(int n,float* a_inOut);
}

#endif
