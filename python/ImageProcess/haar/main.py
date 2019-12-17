import numpy as np
from PIL import Image

img = Image.open("../image/test.jpg").convert("L")#.resize((120,60))
img = np.asarray(img,np.float32)
height,width = img.shape
# out = np.zeros_like(img)

# 以x3为例： 1x3 block
try:
    for ci in range(1,height):
        for cj in range(1,width):
            out = np.zeros_like(img) # haar 特征图
            for i in range(0,height-ci+1,ci):
                for j in range(0,width-cj+1,cj):
                    # block_white1=img[i:i+ci,j-cj:j]
                    # block_white2=img[i:i+ci,j+cj:j+2*cj]
                    block_white1 = np.zeros([ci,cj])
                    block_black=img[i:i+ci,j:j+cj]
                    block_white2 = np.zeros([ci,cj])
                    for k in range(j-cj,j):
                        if k < 0 or k>=width:
                            pass
                        else:
                            block_white1[:,k+cj-j] = img[i:i+ci,k]

                    for k in range(j+cj,j+2*cj):
                        if k < 0 or k>=width:
                            pass
                        else:
                            block_white2[:,k-cj-j] = img[i:i+ci,k]

                    out[i:i+ci,j:j+cj] = block_white1+block_white2-2*block_black

            Image.fromarray(np.clip(out,0,255).astype(np.uint8)).save("haar.jpg")
            exit(0)
except Exception as e:
    print(e)