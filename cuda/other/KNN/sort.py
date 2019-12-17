import numpy as np
import pycuda.autoinit
from pycuda.autoinit import context
import pycuda.driver as cuda
# from pycuda.compiler import DynamicSourceModule
from pycuda.compiler import SourceModule
from pycuda.driver import Stream
import time

def swap(a,b):
    tmp = a
    a = b
    b = tmp
    return a,b


# 选择排序
def cpu_select_sort(result,new_train_y):
    shape = result.shape

    for i in range(shape[0]):
        tmp_result = result[i]
        tmp_new_train_y = new_train_y[i]
        for j in range(shape[1]-1):
            for k in range(j+1,shape[1]):
                # 从小到大排序
                if tmp_result[j]>tmp_result[k]:
                    tmp_result[j],tmp_result[k]=swap(tmp_result[j],tmp_result[k])
                    tmp_new_train_y[j],tmp_new_train_y[k]=swap(tmp_new_train_y[j],tmp_new_train_y[k])

    return result,new_train_y


# 冒泡排序
def cpu_bubble_sort(data:np.array):
    for i in range(len(data)-1): # 循环的次数
        # 每次从后往前依次比较（选择排序是从前往后比）
        for j in range(len(data)-1,i,-1):
            if data[j]<data[j-1]:
                data[j],data[j-1] = swap(data[j],data[j-1])

    return data


# 奇偶排序
def cpu_odd_even_sort(data:np.array):
    for i in range(len(data)-1):
        for j in range(i%2,len(data)-1,2):
            if data[j]>data[j+1]:
                data[j] , data[j + 1] = swap(data[j] , data[j + 1])

    return data


# 归并排序(使用到递归算法) 实现有问题 参考:mergeSort.cpp
def cpu_merge_sort(data:np.array,lindex:int,rindex:int,temp:np.array):
    if lindex<rindex:
        mid = (lindex+rindex)//2
        cpu_merge_sort(data,lindex,mid,temp) # 递归归并左边元素
        cpu_merge_sort(data,mid+1,rindex,temp) # 递归归并右边元素
        data=mergeArray(data, lindex, mid, rindex, temp) # 再将二个有序数列合并

        return data

def mergeArray(data:np.array,lindex:int,mid:int,rindex:int,temp:np.array):
    i = lindex
    j = mid+1
    m = mid
    n = rindex
    k = 0

    while i<=m and j <= n:
        if data[i] <= data[j]:
            temp[k] = data[i]
            k += 1
            i += 1
        else:
            temp[k] = data[j]
            k += 1
            j += 1

    while i<=m:
        temp[k] = data[i]
        k += 1
        i += 1

    while j<=n:
        temp[k] = data[j]
        k += 1
        j += 1

    for i in range(k):
        data[lindex+i] = temp[i]

    return data

def gpu_sort(data:np.array):
    mod = SourceModule("""
        __device__ void swap(float &a,float &b)
        {
            float tmp = a;
            a = b;
            b = tmp;
        }
        
        // 选择排序 按batch=1
        __global__ void sort(float *data)
        {
            //int bx = blockIdx.x;
            int len_data = gridDim.x;
            for(int i =0;i<len_data-1;++i)
            {
                for(int j=i+1;j<len_data;++j)
                {
                    if(data[i]>data[j])
                        swap(data[i],data[j]);
                }
            }   
        }
        
        // 选择排序 -按batch>1
        __global__ void sort2(float *data,int *shape)
        {
            int bx = blockIdx.x;
            // int height = shape[0];
            int width = shape[1];
            for(int i =0;i<width-1;++i)
            {
                for(int j=i+1;j<width;++j)
                {
                    if(data[bx*width+i]>data[bx*width+j])
                        swap(data[bx*width+i],data[bx*width+j]);
                }
            }   
        }
        
        // 奇偶排序
        __global__ void odd_even_sort(float *data)
        {
            int bx = blockIdx.x;
            int by = blockIdx.y;
            
            int idx = by * (gridDim.x+1) + bx;
            
            // if (bx+1>=gridDim.x) return;
            
            for (int i =0;i<=gridDim.x;++i)
            {          
                if(i%2==bx%2 && data[idx]>data[idx+1])
                {
                    swap(data[idx],data[idx+1]);
                }
                
                // __syncthreads(); // 同步
            }   
        }
        
        // 归并排序
        /*
        __global__ void mergesort(float *data,float *tmp,int *shape)
        {
             int tx = threadIdx.x;  
             int bx = blockIdx.x; 
             int idx = tx + bx * blockDim.x;
             if(idx>=shape[0]) return;
             
             // 做一次两两排序 相邻两个都有序，再按每4个，8个，16个... 直到合并所有
             if(idx%2==0 && data[idx]>data[idx+1] && idx+1 < shape[0])
             {
                swap(data[idx],data[idx+1]);
             }
             
             for(int i=2;i<=shape[0]/2;++i)
             {
                for(int j=0;j<i*2;++j)
                {
                    
                }
             }
        } 
        */
        
        __device__ void odd_even_sort_sm(float *sdata)
        {
            int tx = threadIdx.x;
            
            for (int i =0;i<blockDim.x;++i)
            {          
                if(i%2==tx%2 && sdata[tx]>sdata[tx+1] && tx+1<blockDim.x)
                {
                    swap(sdata[tx],sdata[tx+1]);
                }
                
                __syncthreads(); // 同步
            }   
        }
        
        __global__ void mergesort(float *data,int *shape)
        {
            // 先让每个block有序，再合并每个block
            extern __shared__ float sdata[];//共享变量声明
            int tx = threadIdx.x;  
            int bx = blockIdx.x; 
            int idx = tx + bx * blockDim.x;
            if(idx>=shape[0]) return;
            // 全局变量写入到共享变量
            sdata[tx] = data[idx];
            __syncthreads(); // 同步
            // 每个block使用奇偶排序
            odd_even_sort_sm(sdata);
            __syncthreads(); // 同步
            // 共享全局变量写入全局变量
            data[idx] = sdata[tx];
            
            // 合并block    
        }
        
    """)

    g_data = cuda.to_device(data)
    data_shape = data.shape
    g_shape = cuda.to_device(np.asarray(data_shape,np.int32))
    tmp = np.zeros_like(data)
    g_tmp = cuda.to_device(tmp)

    # func = mod.get_function("sort")
    # func(g_data,grid=(len(data),1,1),block=(1,1,1))

    # func = mod.get_function("sort2")
    # g_shape = cuda.to_device(np.asarray(data.shape,np.int32))
    # func(g_data,g_shape, grid=(len(data), 1, 1), block=(1, 1, 1))

    # data_shape = data.shape
    # func = mod.get_function("odd_even_sort")
    # func(g_data, grid=(data_shape[1]-1, data_shape[0], 1), block=(1, 1, 1))


    func = mod.get_function("mergesort")
    block = (16, 1, 1)
    grid = ((len(data)+block[0]-1)//block[0],1,1)
    func(g_data,g_shape, grid=grid, block=block , shared=block[0],stream=Stream(0))

    if grid[0]>1:
        func = mod.get_function("merge")
        func(g_data,g_tmp,g_shape,grid=grid,block=block,shared=0,stream=Stream(0))

    data = cuda.from_device_like(g_data,data)

    return data

def main():
    data = np.asarray([23,20,12,78,45,38,33,49]*4).astype(np.float32)#[None,...]
    # data = np.concatenate([data]*5,0)
    # data = np.random.random([1000,5000]).astype(np.float32)
    # print(cpu_bubble_sort(data))
    # print(cpu_odd_even_sort(data))
    # print(cpu_merge_sort(data,0,4,data))

    start = time.time()
    data1=np.sort(data)
    print(data1)
    print(time.time()-start)

    start = time.time()
    data2 = gpu_sort(data)
    print(data2)
    print(time.time() - start)

if __name__ == '__main__':
    main()