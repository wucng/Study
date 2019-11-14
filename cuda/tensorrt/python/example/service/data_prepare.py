import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import PIL.Image

# load data
def glob_format(path, base_name=False):
    print('--------pid:%d start--------------' % (os.getpid()))
    fmt_list = ('.jpg', '.jpeg', '.png')
    fs = []
    if not os.path.exists(path): return fs
    for root, dirs, files in os.walk(path):
        for file in files:
            item = os.path.join(root, file)
            # item = unicode(item, encoding='utf8')
            fmt = os.path.splitext(item)[-1]
            if fmt.lower() not in fmt_list:
                # os.remove(item)
                continue
            if base_name:
                fs.append(file)  # fs.append(os.path.splitext(file)[0])
            else:
                fs.append(item)
    print('--------pid:%d end--------------' % (os.getpid()))
    return fs

# data for only inference
class MyData(Dataset):
    def __init__(self, base_dir,transform=None):
        super(MyData, self).__init__()
        self.paths = glob_format(os.path.join(base_dir))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = PIL.Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img,path

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

test_transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # 转成0.～1.
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    normalize
])

if __name__=="__main__":
    batch_size = 32#64
    seed = 100
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(seed)
    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}

    base_dir = "/media/wucong/work/practice/data/tomato"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    test_transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # 转成0.～1.
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        normalize
    ])

    test_loader = DataLoader(
        MyData(base_dir, "valid", transform=test_transformations),
        batch_size=batch_size, shuffle=True  # 加上**kwargs导致tensorrt加载数据失败
    )