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

# data
class MyData(Dataset):
    def __init__(self, base_dir, mode="train", target_size=(224, 224),
                 transform=None, target_transform=None, shuffle=False):
        super(MyData, self).__init__()
        self.paths = glob_format(os.path.join(base_dir, mode))

        if shuffle: self._shuttle()

        # self.category_names=category_names
        self.category_names = sorted(os.listdir(os.path.join(base_dir, mode)))
        self.transform = transform
        self.target_transform = target_transform

    def _shuttle(self):
        np.random.shuffle(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = PIL.Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        # target
        target = self.category_names.index(os.path.basename(os.path.dirname(path)))

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

batch_size = 32#64
seed = 100
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(seed)
kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}

base_dir = "/media/wucong/work/practice/data/tomato"

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

train_transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomChoice([
        transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        transforms.RandomCrop(224),
        transforms.CenterCrop(224)
    ]),
    transforms.RandomApply([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
        transforms.RandomAffine(5),
        # transforms.LinearTransformation(),
        # transforms.Grayscale()
    ]),
    # Imgaug(),

    transforms.ToTensor(),  # 转成0.～1.
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # -1.~1.
    normalize
])

test_transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # 转成0.～1.
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    normalize
])

train_loader = DataLoader(
    MyData(base_dir, "train", transform=train_transformations),
    batch_size=batch_size, shuffle=True, **kwargs
)

# test_loader = DataLoader(
#     MyData(base_dir, "valid", transform=test_transformations),
#     batch_size=batch_size, shuffle=True, **kwargs
# )

test_loader = DataLoader(
    MyData(base_dir, "valid", transform=test_transformations),
    batch_size=batch_size, shuffle=True  # 加上**kwargs导致tensorrt加载数据失败
)
