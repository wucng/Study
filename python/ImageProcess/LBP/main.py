"""
图像大小：120x120 (像素)
block大小: 10x10 (cell)
cell大小：12x12 (像素)
每个cell 统计 0～255 像素值 直方图
最后特征：10x10x256
"""

import numpy as np
from PIL import Image

img = Image.open("../image/test.jpg").convert("L").resize((120,60))
img = np.asarray(img,np.uint8)
height,width = img.shape
out = np.zeros_like(img)
kernel_size = (3,3)

for i in range(height):
    for j in range(width):
        tmp = np.zeros(kernel_size, np.uint8)
        for ki in range(kernel_size[0]):
            for kj in range(kernel_size[1]):
                cur_i = i-kernel_size[0]//2+ki
                cur_j = j-kernel_size[1]//2+kj
                if cur_i<0 or cur_i>=height or cur_j<0 or cur_j>=width:
                    pixel = 0
                else:
                    pixel = img[cur_i,cur_j]

                if pixel > img[i,j]:
                    tmp[ki,kj] =1

        # 先转二进制数(去除中心点的值)
        tmp = tmp.flatten()
        binary_number=[c for i,c in enumerate(tmp) if i!=4]

        # binary_number = [tmp[0],tmp[1],tmp[2],tmp[5],tmp[8],tmp[7],tmp[6],tmp[3]]
        # 再转成十进制
        sum = 0
        for i,c in enumerate(binary_number):
            sum += c*2**(len(binary_number)-1-i)

        out[i,j] = sum

Image.fromarray(np.clip(out,0,255).astype(np.uint8)).save("lbp.jpg")
