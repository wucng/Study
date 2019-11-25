#include "function.h"

int main(int argc,char *argv[])
{   
    int nums=5,flag=0;
    if(argc>2)
    {
        nums=atoi(argv[1]);
        flag=atoi(argv[2]);
    }
    Data<FLOAT> data;
    data.allocate(nums,flag);
    exec_vector_add<FLOAT>(data);
    //print
    data.pprint();

    return 0;
}