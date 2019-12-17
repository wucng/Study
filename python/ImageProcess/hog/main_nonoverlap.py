import numpy as np
from PIL import Image

img = Image.open("../image/test.jpg").convert("L")
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

# nonoverlap(没有重叠)
# 行人检测的最佳参数设置
cell= (6,6)
block =(3,3)
num_bin = 9
radian_bin = 2*np.pi/num_bin # 弧度
y_piexl =block[0]*cell[0]
x_piexl =block[1]*cell[1]
num_block = (height//y_piexl,width//x_piexl)
features = np.zeros([num_block[0],num_block[1],block[0],block[1],num_bin])

try:
    for i in range(num_block[0]*y_piexl):
        for j in range(num_block[1]*x_piexl):
            # 第几个block
            block_i = i // y_piexl
            block_j = j // x_piexl

            # 第几个cell
            cell_i=(i%y_piexl)//cell[0]
            cell_j=(j%x_piexl)//cell[1]

            gradValue = gradscal_direct[i, j] if gradscal_direct[i, j] >= 0 else 2 * np.pi + gradscal_direct[i, j]
            features[block_i,block_j,cell_i,cell_j,int(gradValue//radian_bin)]+=1
except:
    print(block_i,block_j,cell_i,cell_j,int(gradValue//radian_bin))

# try:
#     for i in range(num_block[0]):
#         for j in range(num_block[1]):
#             for ci in range(i*y_piexl,(i+1)*y_piexl):
#                 for cj in range(j*x_piexl,(j+1)*x_piexl):
#                     gradValue = gradscal_direct[ci, cj] if gradscal_direct[ci, cj]>=0 else 2*np.pi+gradscal_direct[ci, cj]
#                     features[i,j,(ci-i*y_piexl)//cell[0],(cj-j*x_piexl)//cell[1],int(gradValue//radian_bin)]+=1
# except Exception as e:
#     print(e)

print(features.reshape(-1,block[0]*block[1]*num_bin))
