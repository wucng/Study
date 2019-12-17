"""
python解析mnist数据，保存成npz文件
http://yann.lecun.com/exdb/mnist/
https://blog.csdn.net/lcczzu/article/details/88873156
https://www.jianshu.com/p/e7c286530ab9
"""
import numpy as np
import os
import struct
from tqdm import tqdm
import sys

def parse_mnist_images(data_path):
    """
    :return:
    """

    with open(os.path.join(data_path),'rb') as fp:
        bin_data = fp.read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols),np.uint8)
    for i in tqdm(range(num_images)):
        # if (i + 1) % 10000 == 0:
        #     print('已解析 %d' % (i + 1) + '张')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset),np.uint8).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)

    return images

def parse_mnist_labels(data_path):
    # 读取二进制数据
    bin_data = open(data_path, 'rb').read()
    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images,np.uint8)
    for i in tqdm(range(num_images)):
        # if (i + 1) % 10000 == 0:
        #     print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels

def mnist2npz(data_path):
    train_images_path = os.path.join(data_path,"train-images-idx3-ubyte")
    train_labels_path = os.path.join(data_path,"train-labels-idx1-ubyte")
    test_images_path = os.path.join(data_path,"t10k-images-idx3-ubyte")
    test_labels_path = os.path.join(data_path,"t10k-labels-idx1-ubyte")

    tain_data={}
    test_data={}

    images = parse_mnist_images(train_images_path)
    labels = parse_mnist_labels(train_labels_path)

    tain_data["images"] = images
    tain_data["labels"] = labels
    tain_data["samples"] = len(labels)
    tain_data["dtype"] = "uint8"
    tain_data["max"] = 255

    np.savez("train.npz", **tain_data)

    # --------------------------------------------

    images = parse_mnist_images(test_images_path)
    labels = parse_mnist_labels(test_labels_path)

    test_data["images"] = images
    test_data["labels"] = labels
    test_data["samples"] = len(labels)
    test_data["dtype"] = "uint8"
    test_data["max"] = 255

    np.savez("test.npz", **test_data)



def main(argv):
    data_path = "/media/wucong/work/practice/work/data/mnist"
    if len(argv)>1:
        data_path = argv[1]
    mnist2npz(data_path)

if __name__=="__main__":
    main(sys.argv)
