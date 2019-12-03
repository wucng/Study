"""
# 下载链接：https://www.kaggle.com/noulam/tomato
# 描述：分为训练集与验证集，共10个类别，每张图片大小都为256 x 256

将数据写成这种格式：
path label type
airplanes/image_0316.jpg 0 train
cannon/image_0029.jpg 61 test
"""
import csv
import os
from tqdm import tqdm

# load data
def glob_format(path,base_name = False):
    print('--------pid:%d start--------------' % (os.getpid()))
    fmt_list = ('.jpg', '.jpeg', '.png')
    fs = []
    if not os.path.exists(path):return fs
    for root, dirs, files in os.walk(path):
        for file in files:
            item = os.path.join(root, file)
            # item = unicode(item, encoding='utf8')
            fmt = os.path.splitext(item)[-1]
            if fmt.lower() not in fmt_list:
                # os.remove(item)
                continue
            if base_name:fs.append(file)  # fs.append(os.path.splitext(file)[0])
            else:fs.append(item)
    print('--------pid:%d end--------------' % (os.getpid()))
    return fs

def write2txt(train_path,fp,category=[],type="train"):
    train_data = glob_format(train_path)
    for path in tqdm(train_data):
        label_name=os.path.basename(os.path.dirname(path))
        fp.write(path+" "+str(category.index(label_name))+" "+type+"\n")

if __name__=="__main__":
    train_path = "/media/wucong/work/practice/data/lara2018/classifier/dataset/symptom/train"
    test_path = "/media/wucong/work/practice/data/lara2018/classifier/dataset/symptom/test"

    category = list(sorted(os.listdir(train_path)))
    fp = open("info.txt", 'w')

    write2txt(train_path,fp,category,"train")
    write2txt(test_path,fp,category,"test")

    fp.close()
