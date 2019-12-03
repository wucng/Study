# https://www.cnblogs.com/king-lps/p/9031568.html
import cv2
import numpy as np
import random
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time

# boxes=[[10,10,120,120],[20,20,100,100],[30,30,90,90],[15,18,210,220],[10,10,120,120]]
# scores=[0.9,0.87,0.85,0.79,0.75]
"""
boxes = []
scores = []
for i in range(5000):
    boxes.append([random.randint(0,100),random.randint(0,100),
                  random.randint(150,300),random.randint(150,300)])
    scores.append(random.random())
"""
boxes = np.array([[100, 100, 210, 210, 0.72],
                  [250, 250, 420, 420, 0.8],
                  [220, 220, 320, 330, 0.92],
                  [100, 100, 210, 210, 0.72],
                  [230, 240, 325, 330, 0.81],
                  [220, 230, 315, 340, 0.9]])
scores=boxes[...,-1]
index = scores.argsort()[::-1] #从大到小排序
boxes = boxes[...,:4][index].astype(int)


img=np.zeros([450,450,3],np.uint8)

def overArea(box1,box2):
    """
    :param box1: [x1,y1,x2,y2]
    :param box2: [x1,y1,x2,y2]
    :return:
    """
    x11,y11,x12,y12=box1
    x21,y21,x22,y22=box2
    x1=max(x11,x21)
    y1=max(y11,y21)
    x2=min(x12,x22)
    y2=min(y12,y22)
    w=x2-x1
    h=y2-y1
    if w<=0 or h<=0:
        intersectArea=0
    else:
        intersectArea=w*h

    unionArea=(x12-x11)*(y12-y11)+(x22-x21)*(y22-y21)-intersectArea

    return intersectArea/unionArea

# nms
def nms(boxes,threshold=0.5):
    """
    # 1.先按scores过滤分数低的,过滤掉分数小于conf_thres
    # 2.类别一样的按nms过滤，如果Iou大于nms_thres,保留分数最大的,否则都保留
    :param boxes: 按scores从大到小排序，必须是同一类别
    :param scores:
    :param threshold:
    :return:
    """
    mark=[1]*len(boxes) # 标记1 保留 标记0 舍弃
    for i in range(len(boxes)-1):
        if mark[i]!=1:continue
        for j in range(i+1,len(boxes)):
            # 计算重叠面积
            if mark[j] == 0: continue
            if overArea(boxes[i],boxes[j])>threshold:
                mark[j]=0
        # break
    return mark

def gpu_nms(boxes,threshold=0.5):
    """"""
    # Executing a Kernel
    mod = SourceModule("""
      __device__ bool overArea(float x11,float y11,float x12,float y12,
                                float x21,float y21,float x22,float y22,
                                float threshold)
      {
            float intersectArea=0.0f;
            float x1=fmaxf(x11,x21);  
            float y1=fmaxf(y11,y21);  
            float x2=fminf(x12,x22);  
            float y2=fminf(y12,y22);  
            if ((x2-x1)<=0 || (y2-y1)<=0)
            {}
            else
            {
                intersectArea=(x2-x1)*(y2-y1);
            }
            float unionArea=(x12-x11)*(y12-y11)+(x22-x21)*(y22-y21)-intersectArea;
            
            return (intersectArea/unionArea>threshold);
      }
      
      
      __global__ void nms(float *boxes,int *mark,float *threshold,int *boxesshape)
      {
           int tx = threadIdx.x; 
           // int ty = threadIdx.y;
           int idx = tx + blockIdx.x*blockDim.x;
           int b_rows = boxesshape[0];
           int b_cols = boxesshape[1];
           int new_idx = 0;
           
           for(int i=0;i<b_rows-1;++i) // 后一次循环必须使用到前一步的结果，强行将这个串行循环拆开，结果出错
           {   
               if (mark[i]!=1) continue;
               new_idx = i+1+idx;
               if (new_idx >=b_rows || mark[new_idx]==0) return;
               if (overArea(boxes[i*b_cols+0],boxes[i*b_cols+1],boxes[i*b_cols+2],boxes[i*b_cols+3],
               boxes[new_idx*b_cols+0],boxes[new_idx*b_cols+1],boxes[new_idx*b_cols+2],boxes[new_idx*b_cols+3],threshold[0])
               )
                    mark[new_idx]=0; 
           }
      }
      
      __global__ void sm_nms(float *boxes,int *mark,float *threshold,int *boxesshape)
      {
           __shared__ float sdatas[6*4];
           int tx = threadIdx.x; 
           // int ty = threadIdx.y;
           int idx = tx + blockIdx.x*blockDim.x;
           int b_rows = boxesshape[0];
           int b_cols = boxesshape[1];
           int new_idx = 0;
           
           if (tx>=b_rows) return;
           for (int i =0;i<b_cols;++i)
                sdatas[tx*b_cols+i]= boxes[tx*b_cols+i];
           __syncthreads();
           
           for(int i=0;i<b_rows-1;++i) // 后一次循环必须使用到前一步的结果，强行将这个串行循环拆开，结果出错
           {   
               if (mark[i]!=1) continue;
               new_idx = i+1+idx;
               if (new_idx >=b_rows || mark[new_idx]==0) return;
               if (overArea(sdatas[i*b_cols+0],sdatas[i*b_cols+1],sdatas[i*b_cols+2],sdatas[i*b_cols+3],
               sdatas[new_idx*b_cols+0],sdatas[new_idx*b_cols+1],sdatas[new_idx*b_cols+2],sdatas[new_idx*b_cols+3],threshold[0])
               )
                    mark[new_idx]=0; 
           }
      }
      """)

    boxes=np.asarray(boxes,np.float32)
    mark = np.ones([len(boxes)],np.int32)   # 标记1 保留 标记0 舍弃
    g_boxes = cuda.to_device(boxes)
    g_boxeshape = cuda.to_device(np.asarray(boxes.shape,np.int32))
    g_mark = cuda.to_device(mark)
    g_thres = cuda.to_device(np.asarray([threshold],np.float32))
    # func = mod.get_function("nms")
    func = mod.get_function("sm_nms")
    block=(256,1,1)
    grid=((len(boxes)+256-1)//256,1,1)

    func(g_boxes,g_mark,g_thres,g_boxeshape, grid=grid, block=block)  # 调用核函数

    mark=cuda.from_device(g_mark,mark.shape,mark.dtype)

    return mark

start=time.time()
# mark=nms(boxes,0.5)
mark=gpu_nms(boxes,0.5) # [1 0 1 1 0]
print(time.time()-start)
print(np.sum(np.asarray(mark)))

for i,box in enumerate(boxes):
    if mark[i]==1:
        img=cv2.rectangle(img,tuple(box[:2]),tuple(box[2:]),(0,0,255))

cv2.imshow("test",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
