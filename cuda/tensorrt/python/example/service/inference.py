"""
对Python3 pyc 文件的使用详解:https://www.jb51.net/article/156384.htm
将.py文件编译成.pyc文件，.pyc使用方式与.py一样
"""
from data_prepare import MyData,test_transformations
from model import MyModel,predict,category
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import shutil
import os


batch_size=32
num_classes = 10
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = MyModel(num_classes, "resnet18", 512, False, 0.0)
model.to(device)
state_dict = torch.load("./model.pt")
model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict()})

# base_dir = "/media/wucong/work/practice/data/tomato"
base_dir = "/media/wucong/work/practice/data/test"
test_loader=DataLoader(MyData(base_dir,test_transformations),batch_size)

predict(model,device,test_loader,category)