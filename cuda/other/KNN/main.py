"""
使用pycuda实现K最近邻算法
"""
import numpy as np
import pycuda.autoinit
from pycuda.autoinit import context
import pycuda.driver as cuda
# from pycuda.compiler import DynamicSourceModule
from pycuda.compiler import SourceModule
from pycuda.driver import Stream
# from PIL import Image
# from sort import cpu_sort


def loadData():
    train_data = np.load("train.npz")
    test_data = np.load("test.npz")

    train_x = 1.0*train_data["images"]/train_data["max"] # 0.~1.0
    train_y = train_data["labels"]

    test_x = 1.0 * test_data["images"] / test_data["max"]  # 0.~1.0
    test_y = test_data["labels"]

    return (train_x,train_y),(test_x,test_y)

def knn(train_x,train_y,test_x,test_y):
    mod = SourceModule("""
        __global__ void knn(float *query,float *data,float *result)
        {
            int bx = blockIdx.x;
            int by = blockIdx.y;
            int idx = bx + by * gridDim.x;
            
            float value = 0.0f;
            
            value = (query[idx] - data[idx])*(query[idx] - data[idx]);
            
            atomicAdd(&result[0],value);
            
            //if(idx==0)
            //    result[0] = sqrtf(result[0]);   
        }
    """)

    n=7
    func = mod.get_function("knn")

    train_x = train_x.astype(np.float32).reshape(-1,28*28)
    test_x = test_x.astype(np.float32).reshape(-1,28*28)

    # g_train_x = cuda.to_device(train_x)
    # g_test_x = cuda.to_device(test_x)
    # g_train_x.free()
    # g_test_x.free()

    result = np.zeros([len(test_x),len(train_x)],np.float32)
    pred = np.zeros([len(test_x)],np.uint8)

    for i in range(len(test_x)):
        g_test = cuda.to_device(test_x[i])
        for j in range(len(train_x)):
            g_train = cuda.to_device(train_x[j])
            g_result = cuda.to_device(result[i,j])
            func(g_test,g_train,g_result,grid=(28,28,1),block=(1,1,1),shared=0,stream=Stream(0))
            result[i, j]=cuda.from_device_like(g_result,result[i, j])
            g_train.free()
            g_result.free()

        g_test.free()

        # 每一行排序得到最终结果
        tmp=train_y[np.argsort(result[i]).tolist()][:n]
        pred[i] = stateLabel(tmp)

    # 计算精度
    print("精度：",np.sum(pred==test_y)/len(test_y))


def knn2(train_x, train_y, test_x, test_y):
    mod = SourceModule("""
        __global__ void knn(float *query,float *data,float *result)
        {
            int bx = blockIdx.x;
            int by = blockIdx.y;
            
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            
            int bid = bx + by * gridDim.x;
            int tid = tx + ty * blockDim.x;
            // int idx = tid + bid * blockDim.y * blockDim.x;
            
            int query_idx = tid + by * blockDim.y * blockDim.x;
            int data_idx = tid + bx * blockDim.y * blockDim.x;
            
            float value = 0.0f;

            value = (query[query_idx] - data[data_idx])*(query[query_idx] - data[data_idx]);

            atomicAdd(&result[bid],value);

            //if(idx==0)
            //    result[0] = sqrtf(result[0]);   
        }
    """)

    n = 13
    func = mod.get_function("knn")

    train_x = train_x.astype(np.float32).reshape(-1, 28 * 28)
    test_x = test_x.astype(np.float32).reshape(-1, 28 * 28)


    result = np.zeros([len(test_x), len(train_x)], np.float32)
    pred = np.zeros([len(test_x)], np.uint8)


    g_test = cuda.to_device(test_x)

    g_train = cuda.to_device(train_x)
    g_result = cuda.to_device(result)
    func(g_test, g_train, g_result, grid=(len(train_x), len(test_x), 1), block=(28, 28, 1), shared=0, stream=Stream(0))
    result = cuda.from_device_like(g_result, result)

    g_train.free()
    g_result.free()
    g_test.free()

    # 每一行排序得到最终结果
    # for i in range(len(test_x)):
    #     tmp = train_y[np.argsort(result[i]).tolist()][:n]
    #     pred[i] = stateLabel(tmp)

    index = np.argsort(result)[:, :n]
    for i in range(len(test_x)):
        tmp = train_y[index[i]]
        pred[i] = stateLabel(tmp)

    # 计算精度
    print("精度：", np.sum(pred == test_y) / len(test_y))

def stateLabel(tmp:np.array)->int:
    label =0
    sum = 0
    for i in range(10):
        if sum < np.sum(tmp==i):
            sum = np.sum(tmp==i)
            label = i

    return label


def main():
    (train_x,train_y),(test_x,test_y) = loadData()
    # print(train_x.shape)
    # print(train_y.shape)
    # print(test_x.shape)
    # print(test_y.shape)

    # knn(train_x,train_y,test_x,test_y)
    # knn(train_x[:5000],train_y[:5000],test_x[:1000],test_y[:1000])
    knn2(train_x[:10000],train_y[:10000],test_x[:1000],test_y[:1000])
    # knn2(train_x[:20],train_y[:20],test_x[:20],test_y[:20])

if __name__=="__main__":
    main()
