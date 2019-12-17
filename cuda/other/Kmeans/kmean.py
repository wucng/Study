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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import manifold
import random

def loadData():
    train_data = np.load("train.npz")
    test_data = np.load("test.npz")

    train_x = 1.0*train_data["images"]/train_data["max"] # 0.~1.0
    train_y = train_data["labels"]

    test_x = 1.0 * test_data["images"] / test_data["max"]  # 0.~1.0
    test_y = test_data["labels"]

    return (train_x,train_y),(test_x,test_y)

def visual_feature_PCA(feature,y_pred):
    """
    :param feature: [bs.m]
    :param y_pred: [bs,]
    :return:
    """
    # pca
    pca = PCA(n_components=2)
    pca.fit(feature)
    X_new = pca.transform(feature)
    plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y_pred)
    plt.show()

def visual_feature_TSNE(feature,y_pred):
    """
    :param feature: [bs.m]
    :param y_pred: [bs,]
    :return:
    """
    # "t-SNE"
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(feature)
    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y_pred[i]), color=plt.cm.Set1(y_pred[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.show()

def distance(data1,data2):
    return np.sum((data1-data2)**2)

def kmean(data,meanSize=10):
    shape = data.shape
    # 随机选10个点作为初始化点
    means=[]
    for i in range(0,shape[0],shape[0]//meanSize):
        means.append(data[random.randint(i,i+shape[0]//meanSize)])

    iters=0
    err = 100.
    while err>1.e-3 or iters<200:
        data_dict={}
        labels = []
        # 迭代，计算每个样本到这些means的距离，排序，分到距离最小的
        for i in range(shape[0]):
            tmp = []  # 存储每个样本到样本均值的距离
            for j in range(meanSize):
                tmp.append(distance(data[i],means[j]))
            label=np.argsort(tmp)[0]
            labels.append(label)
            if label not in data_dict:
                data_dict[label]=[]
            data_dict[label].append(data[i])

        # 重新计算mean
        new_mean=[]
        for j in range(meanSize):
            new_mean.append(np.mean(np.asarray(data_dict[j]),0))

        # 计算新旧means的误差
        err = np.sqrt(np.sum((np.asarray(means)-np.asarray(new_mean))**2))
        iters+=1

        # 更新means
        means = new_mean

    return labels,err,iters

if __name__=="__main__":
    (train_x, train_y), (test_x, test_y)= loadData()
    train_x = train_x.reshape(-1,28*28)
    test_x = test_x.reshape(-1,28*28)

    # visual_feature_TSNE(test_x[:200],test_y[:200])
    labels,err,iters=kmean(test_x[:200])
    print(err,iters)
    visual_feature_TSNE(test_x[:200], labels)