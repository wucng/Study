#include "common.h"
#include "cudaFunction.cuh"
typedef float FLOAT;

template <typename T>
extern void exec_vector_add(Data<T>& data);
