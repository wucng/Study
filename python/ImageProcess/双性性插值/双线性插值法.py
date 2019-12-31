"""
图像处理之双线性插值法
https://blog.csdn.net/qq_37577735/article/details/80041586
"""
import cv2
import numpy as np

# 双线性插值法(灰度图)
def bilinear_interpolation(img:np.array,newSize:tuple=(224,224))->np.array:
    height, width = img.shape
    new_h,new_w = newSize
    new_img = np.zeros((new_h, new_w), np.float32)
    print("height:%d,width:%d" % (height, width))
    for i in range(new_h):
        for j in range(new_w):
            # y = i*height / new_h
            # x = j*width / new_w
            # or matlab和openCV的做法
            y = (i + 0.5) * height / new_h - 0.5
            x = (j + 0.5) * width / new_w - 0.5

            # 双线性插值
            x1 = int(np.floor(x))  # 向下取整
            y1 = int(np.floor(y))
            x2 = int(np.ceil(x))  # 向上取整
            y2 = int(np.ceil(y))
            if x1<0:x1=0
            if y1<0:y1=0
            if x2>=width:x2=width-1
            if y2>=height:y2=height-1

            new_img[i, j] = img[y1, x1] * (1 - abs(x1 - x)) * (1 - abs(y1 - y)) + \
                            img[y1, x2] * (1 - abs(x2 - x)) * (1 - abs(y1 - y)) + \
                            img[y2, x2] * (1 - abs(x2 - x)) * (1 - abs(y2 - y)) + \
                            img[y2, x1] * (1 - abs(x1 - x)) * (1 - abs(y2 - y))

    return new_img

# 双线性插值法(彩色图，多通道图)
def bilinear_interpolation_BGR(img:np.array,newSize:tuple=(224,224))->np.array:
    height, width, channels = img.shape
    new_h,new_w = newSize
    new_img = np.zeros((new_h, new_w,channels), np.float32)
    print("height:%d,width:%d,channels:%d" % (height, width, channels))
    for i in range(new_h):
        for j in range(new_w):
            for c in range(channels):
                # y = i*height / new_h
                # x = j*width / new_w
                # or matlab和openCV的做法
                y = (i + 0.5) * height / new_h - 0.5
                x = (j + 0.5) * width / new_w - 0.5

                # 双线性插值
                x1 = int(np.floor(x))  # 向下取整
                y1 = int(np.floor(y))
                x2 = int(np.ceil(x))  # 向上取整
                y2 = int(np.ceil(y))
                if x1<0:x1=0
                if y1<0:y1=0
                if x2>=width:x2=width-1
                if y2>=height:y2=height-1

                new_img[i, j, c] = img[y1, x1, c] * (1 - abs(x1 - x)) * (1 - abs(y1 - y)) + \
                                img[y1, x2, c] * (1 - abs(x2 - x)) * (1 - abs(y1 - y)) + \
                                img[y2, x2, c] * (1 - abs(x2 - x)) * (1 - abs(y2 - y)) + \
                                img[y2, x1, c] * (1 - abs(x1 - x)) * (1 - abs(y2 - y))

    return new_img

# downsample(下采样，取奇数行和列 或 偶数行和列，尺寸缩减一半)
def downsample(img:np.array,even:bool=True)->np.array:
    # 默认取偶数，间隔取
    height, width, channels = img.shape
    new_h, new_w = height//2, width//2
    new_img = np.zeros((new_h, new_w, channels), np.float32)
    for i in range(height):
        for j in range(width):
            for c in range(channels):
                if even:
                    if i%2==0 and j%2==0:
                        new_img[i // 2, j // 2, c] = img[i, j, c]
                else:
                    if i%2==1 and i%2==1:
                        new_img[i // 2, j // 2, c] = img[i, j, c]

    return new_img

# upsample(上采样，每两个像素（行与列）间插一个值（两个像素的平均值）)
def upsample(img:np.array)->np.array:
    height, width, channels = img.shape
    new_h, new_w = 2*height-1,2*width-1
    new_img = np.zeros((new_h, new_w, channels), np.float32)
    for i in range(new_h):
        for j in range(new_w):
            for c in range(channels):
                if i%2==0 and j%2==0:
                    new_img[i,j,c] = img[i//2,j//2,c]
                else:
                    if i<j:
                        new_img[i, j, c] = (img[i//2,(j-1)//2,c]+img[i//2,(j+1)//2,c])/2
                    elif i>j:
                        new_img[i, j, c] = (img[(i-1)//2,j//2,c]+img[(i+1)//2,j//2,c])/2
                    else:
                        new_img[i, j, c] = (img[(i-1)//2,(j-1)//2,c]+img[(i-1)//2,(j+1)//2,c]+ img[(i+1)//2,(j-1)//2,c]+img[(i+1)//2,(j+1)//2,c])/4

    return new_img

if __name__=="__main__":
    """
    img = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)  # IMREAD_COLOR
    img = bilinear_interpolation(img,(200,200)) # (1080,1920)
    new_img = bilinear_interpolation(img,(400,400))
    """
    img = cv2.imread("test.jpg", cv2.IMREAD_COLOR)  # IMREAD_COLOR
    # img = bilinear_interpolation_BGR(img, (200, 200))  # (1080,1920)
    # new_img = bilinear_interpolation_BGR(img, (400, 400))

    # new_img = downsample(img,False)
    img = bilinear_interpolation_BGR(img, (200, 200))
    new_img = upsample(img)
    cv2.imshow("test",new_img.clip(0,255).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()