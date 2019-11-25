"""
使用cupy加载权重文件，实现分类

# For CUDA 10.0
pip install cupy-cuda100

# For CUDA 10.1
pip install cupy-cuda101
"""
# import numpy as np
import cupy as cp
import tensorflow as tf
import time

# 1、加载npz文件
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

def model(parmas,img):
    """手动解析权重计算img的输出做分类"""
    # x=cp.reshape(img,[-1,28*28*1]).astype('f')
    x=img
    # for key,value in parmas.items():
    x=cp.matmul(x, parmas['dense/kernel:0'])+parmas['dense/bias:0']
    x=relu(x)
    x=cp.matmul(x,parmas['dense_1/kernel:0'])+parmas['dense_1/bias:0']

    # softmax
    return softmax(x)

start=time.time()
x_train, y_train, x_test, y_test=process_dataset()
print(cp.argmax(model(parmas,cp.asarray(x_test[:20],cp.float32).reshape([-1,28*28*1])),1))
print(y_test[:20])
print(time.time()-start)
"""
[7 2 1 0 4 1 4 9 6 9 0 6 9 0 1 5 9 7 3 4]
[7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4]
0.8439
"""
