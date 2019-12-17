import numpy as np
from PIL import Image

img = Image.open("../image/test.jpg").convert("L")
img = np.asarray(img,np.float32)
height,width = img.shape

# RGB-->Gray
# Image.fromarray(np.clip(img,0,255).astype(np.uint8)).save("gray.jpg")

# 标准化gamma空间和颜色空间
gamma = 1.0 # 0.5
img = img**gamma

# Image.fromarray(np.clip(img,0,255).astype(np.uint8)).save("gamma.jpg")