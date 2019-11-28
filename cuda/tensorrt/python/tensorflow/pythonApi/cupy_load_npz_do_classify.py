"""
使用cupy加载权重文件，实现分类
"""
import cupy as cp
import numpy as np
import tensorflow as tf
import time

# 1、加载cpz文件
parmas=cp.load("models/tf_args.npz")
# print(list(parmas.keys()))

def process_dataset():
    # Import the data
    (x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshape the data
    NUM_TRAIN = 60000
    NUM_TEST = 10000
    x_train = cp.reshape(x_train, (NUM_TRAIN, 28, 28, 1))
    x_test = cp.reshape(x_test, (NUM_TEST, 28, 28, 1))
    return x_train, y_train, x_test, y_test

def softmax(x):
    x_exp = cp.exp(x)
    #如果是列向量，则axis=0
    x_sum = cp.sum(x_exp, axis = 1, keepdims = True)
    s = x_exp / x_sum
    return s

def relu(x):
    s = cp.where(x < 0, 0, x)
    return s

def batch_norm(x,gamma,beta,mean,var,esp=1e-3):
    scale = gamma/cp.sqrt(var+esp)
    shift = -mean/cp.sqrt(var+esp)*gamma+beta

    return x*scale+shift

def _conv2d(x,out,weight,bias,x_shape,w_shape):
    """
    x:[28,28,1]
    w:[5,5,1]
    b:[1]
    return : [28,28,1]
    """
    # pixel=0.0
    for m in range(x_shape[0]):
        for n in range(x_shape[1]):
            for i in range(w_shape[0]):
                for j in range(w_shape[1]):
                    cur_row = m-w_shape[0]//2+i
                    cur_col = n-w_shape[1]//2+j
                    for k in range(w_shape[2]):
                        if cur_row<0 or cur_row>=x_shape[0] or cur_col<0 or cur_col>=x_shape[1]:
                            pixel=0
                        else:
                            pixel=x[cur_row,cur_col,k]

                        out[m,n]+=pixel*weight[i,j,k]

            out[m,n]+=bias

    return out

def conv2d(x,weight,bias):
    """
    x:[-1,28,28,1] -->[-1,24,24,20]
    w:[5,5,1,20]
    b:[20,]
    s:1
    p:valid
    """

    x_shape = x.shape
    w_shape = weight.shape
    out = cp.zeros((*x_shape[:-1],w_shape[-1]))
    for i in range(x_shape[0]):
        img=x[i,...]
        for j in range(w_shape[-1]):
            out[i,:,:,j]=_conv2d(img,out[i,:,:,j],weight[...,j],bias[j],x_shape[1:],w_shape[:-1])
    
    # [-1,28,28,1] -->[-1,24,24,20]
    return out[:,2:x_shape[1]-2,2:x_shape[2]-2,:]

def _maxpool(x,out,x_shape):
    """
    x:[24,24]
    out:[12,12]
    s:2x2
    k:2x2
    """
    kernel_h=kernel_w=2
    for m in range(x_shape[0]):
        for n in range(x_shape[1]):
            if (m+n)%2==0 and m+n>0:
                for i in range(kernel_h): # kernel_h
                    for j in range(kernel_w): # kernel_w
                        cur_row = m-kernel_h//2+i
                        cur_col = n-kernel_w//2+j
                        pixel=x[cur_row,cur_col]
                        # 取最大
                        out[m//2,n//2]=max(out[m//2,n//2],pixel)
    
    return out

def maxpool(x):
    """
    x:[-1,24,24,20]-->[-1,12,12,20]
    s:2x2
    k:2x2
    """
    x_shape = x.shape
    out = cp.ones([x_shape[0],x_shape[1]//2,x_shape[2]//2,x_shape[3]],x.dtype)*(-99999)
    for i in range(x_shape[0]):
        for j in range(x_shape[-1]):
            out[i,:,:,j]=_maxpool(x[i,:,:,j],out[i,:,:,j],(x_shape[1],x_shape[2]))

    return out


def model(parmas,img):
    """手动解析权重计算img的输出做分类"""
    x=cp.reshape(img,[-1,28,28,1]).astype(cp.float32)
    x=conv2d(x,parmas["conv2d/kernel:0"],parmas["conv2d/bias:0"])
    x=batch_norm(x,parmas["batch_normalization/gamma:0"],parmas["batch_normalization/beta:0"],
    parmas["batch_normalization/moving_mean:0"],parmas["batch_normalization/moving_variance:0"])
    x=relu(x)
    x=maxpool(x)

    x=conv2d(x,parmas["conv2d_1/kernel:0"],parmas["conv2d_1/bias:0"])
    x=batch_norm(x,parmas["batch_normalization_1/gamma:0"],parmas["batch_normalization_1/beta:0"],
    parmas["batch_normalization_1/moving_mean:0"],parmas["batch_normalization_1/moving_variance:0"])
    x=relu(x)
    x=maxpool(x)

    x = cp.reshape(x,[x.shape[0],-1])

    x=cp.matmul(x,parmas['dense/kernel:0'])+parmas['dense/bias:0']
    x=relu(x)
    x=cp.matmul(x,parmas['dense_1/kernel:0'])+parmas['dense_1/bias:0']

    # softmax
    return softmax(x)

start=time.time()
x_train, y_train, x_test, y_test=process_dataset()
print(cp.argmax(model(parmas,x_test[:20]),1))
print(y_test[:20])
print(time.time()-start)
"""
[7 2 1 0 4 1 4 3 5 7 0 6 7 0 1 5 0 7 3 4]
[7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4]
224.34830594062805
"""
