import numpy as np
from PIL import Image

img = Image.open("../image/test.jpg").convert("L").resize((128,64)) # 64x128
img = np.asarray(img,np.float32)
height,width = img.shape

# gamma image
# gamma = 0.5
# img = img**gamma


# [-1,0,1]梯度算子对原图像做卷积运算，得到x方向的梯度分量gradscalx
gradscalx = np.zeros([height,width],np.float32)
kenerl = np.asarray([-1,0,1],np.float32)
cur_j = 0
for i in range(height):
    for j in range(width):
        sum = 0
        for k in range(len(kenerl)):
            cur_j = j-len(kenerl)//2+k
            if cur_j <0 or cur_j >= width:
                piexl = 0
            else:
                piexl = img[i,cur_j]

            sum += piexl*kenerl[k]

        gradscalx[i,j] = sum

# 用[1,0,-1]^T梯度算子对原图像做卷积运算，得到y方向（竖直方向，以向上为正方向）的梯度分量gradscaly
gradscaly = np.zeros([height,width],np.float32)
kenerl = np.asarray([1,0,-1],np.float32)
cur_i = 0
for i in range(height):
    for j in range(width):
        sum = 0
        for k in range(len(kenerl)):
            cur_i = i-len(kenerl)//2+k
            if cur_i <0 or cur_i >= height:
                piexl = 0
            else:
                piexl = img[cur_i,j]

            sum += piexl*kenerl[k]

        gradscaly[i,j] = sum


# 计算该像素点的梯度大小和方向
# gradscal = np.zeros([height,width],np.float32)
# gradscal_direct = np.zeros([height,width],np.float32)

gradscal = np.sqrt(gradscalx**2+gradscaly**2)
gradscal_direct = np.arctan(gradscalx/(gradscaly+1e-3))

# save
# Image.fromarray(np.clip(gradscal,0,255).astype(np.uint8)).save("gradscal.jpg")

# 将梯度方向360分成9个bin，则每个bin为40
# 行人检测的最佳参数设置是：3×3cell/block、6×6像素/cell、9个直方图通道。则一块的特征数为：3*3*9

# 对于64*128的图像而言，每16*16的像素组成一个cell，每2*2个cell组成一个块，因为每个cell有9个特征，
# 所以每个块内有2*2*9=36个特征，以8个像素为步长，那么，水平方向将有5个扫描窗口，垂直方向将有13个扫描窗口。
# 也就是说，64*128的图片，总共有36*5*13=2340个特征。

# overlap(有重叠)
"""
图像size: 64x128（像素）
每个cell: 16x16（像素）
每个block: 2x2 (cell)  即32x32（像素）
每个cell：9个特征（9个方向）

1.nonoverlap
以2×16=32个像素为步长
block数：(64//32,128//32)=(2,4)
features = np.zeros([2,4,2,2,9]) # 每个cell 统计9个方向

2.overlap
以1×16/2=8个像素为步长
block数计算：
x方向：0～32，8～40,16～48,24～56,32～64 （5个）
y方向：0～32，8～40,16～48,24～56,32～64，40～72,48～80,56～88,64～96,72～104,80～112,88～120,96～128 （13）

(64-32)//8+1=5
(128-32)//8+1=13
features = np.zeros([5,13,2,2,9]) # 每个cell 统计9个方向
"""
# 行人检测的最佳参数设置
# cell= (6,6)
# block =(3,3)
cell= (16,16)
block =(2,2)
num_bin = 9
radian_bin = 2*np.pi/num_bin # 弧度
strides = (cell[0]//2,cell[1]//2) # 步长
y_piexl =block[0]*cell[0]
x_piexl =block[1]*cell[1]

new_height = height//strides[0]*strides[0]
new_width = width//strides[1]*strides[1]

num_block = ((new_height-y_piexl)//strides[0]+1,(new_width-x_piexl)//strides[1]+1)
features = np.zeros([num_block[0],num_block[1],block[0],block[1],num_bin])

try:
    for i in range(0,new_height-y_piexl+1,strides[0]):
        for j in range(0,new_width-x_piexl+1,strides[1]):
            # 第几个block
            block_i = i // strides[0]
            block_j = j // strides[1]

            for ci in range(i,i+y_piexl):
                for cj in range(j, j + x_piexl):
                    # 第几个cell
                    cell_i = (ci-i) // cell[0]
                    cell_j = (cj-j) // cell[1]
                    gradValue = gradscal_direct[ci, cj] if gradscal_direct[ci, cj] >= 0 else 2 * np.pi + gradscal_direct[ci, cj]
                    features[block_i,block_j,cell_i,cell_j,int(gradValue//radian_bin)]+=1
except Exception as e:
    print(e)

print(features.reshape(-1,block[0]*block[1]*num_bin))
# print(features[-1,-1,...])