"""
完整版，所有模块封装到类里
"""
from __future__ import print_function
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import PIL.Image
from collections import OrderedDict
import math
import matplotlib.pyplot as plt
import json

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

# model resnet
class MyResnet(nn.Module):
    def __init__(self, num_classes, model_name="resnet101", backbone_size=2048, pretrained=False, droprate=0.0,
                 device="cpu"):
        super(MyResnet, self).__init__()
        self.pretrained = pretrained

        _model = torchvision.models.resnet.__dict__[model_name](pretrained=pretrained)
        self.backbone = nn.Sequential(OrderedDict([
            ('conv1', _model.conv1),
            ('bn1', _model.bn1),
            ('relu1', _model.relu),
            ('maxpool1', _model.maxpool),

            ("layer1", _model.layer1),
            ("layer2", _model.layer2),
            ("layer3", _model.layer3),
            ("layer4", _model.layer4),
        ]))
        self._conv1 = nn.Sequential(
            nn.Dropout(droprate),
            nn.Conv2d(backbone_size, num_classes, 1, 1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.avgpool = nn.AvgPool2d(1,stride=7,padding=0))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self._conv1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x

# model mnasnet
class MyMnasnet(nn.Module):
    def __init__(self, num_classes, model_name="mnasnet0_5", backbone_size=1280, pretrained=False, dropout=0.0,
                 device="cpu"):
        super(MyMnasnet, self).__init__()
        self.pretrained = pretrained

        _model = torchvision.models.mnasnet.__dict__[model_name](pretrained=pretrained)
        self.backbone = nn.Sequential(OrderedDict([
            ('layers', _model.layers)
        ]))

        self.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True),
                                        nn.Linear(backbone_size, num_classes))

    def forward(self, x):
        x = self.backbone(x)
        # Equivalent to global avgpool and removing H and W dimensions.
        x = x.mean([2, 3])
        return self.classifier(x)

# model densenet
class MyDensenet(nn.Module):
    def __init__(self, num_classes, model_name="resnet101", backbone_size=2048, pretrained=False, droprate=0.0,
                 device="cpu"):
        super(MyDensenet, self).__init__()
        self.pretrained = pretrained

        _model = torchvision.models.densenet.__dict__[model_name](pretrained=pretrained)
        self.backbone = nn.Sequential(OrderedDict([
            ('features', _model.features)
        ]))

        # Linear layer
        self.classifier = nn.Sequential(nn.Dropout(p=droprate, inplace=True),
                                        nn.Linear(backbone_size, num_classes))

    def forward(self, x):
        features = self.backbone(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

class LossFunc(object):
    """
    自定义loss 函数
    """
    def __init__(self,num_classes=10,reduction="mean",esp = 1e-5):
        self.num_classes = num_classes
        self.reduction = reduction
        self.esp = esp

    def focal_cross_entropy(self, input, target, alpha=None, gamma=2):
        # 转one-hot
        if target.ndim == 1:
            target = torch.eye(self.num_classes, self.num_classes,device=target.device)[target]

        # loss = -target*torch.log_softmax(input,-1)
        if alpha is None:
            alpha = torch.ones(self.num_classes, dtype=target.dtype, device=target.device)
        else:  # alpha = [1,0.5,...], len(alpha) = num_classes
            alpha = torch.as_tensor(alpha, dtype=target.dtype, device=target.device)
        input = torch.softmax(input, -1)
        loss = -alpha * (1 - input) ** gamma * target * torch.log(input)

        if self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "mean":
            loss = torch.mean(loss)
        else:
            raise ("reduction must in [sum,mean]")

        return loss

    def focal_mse(self, input, target, alpha=None, gamma=2):
        # 转one-hot
        if target.ndim == 1:
            target = torch.eye(self.num_classes, self.num_classes,device=target.device)[target]
        input = torch.softmax(input, -1)
        # loss = (input-target)**2
        if alpha is None:
            alpha = torch.ones(self.num_classes, dtype=target.dtype, device=target.device)
        else:  # alpha = [1,0.5,...], len(alpha) = num_classes
            alpha = torch.as_tensor(alpha, dtype=target.dtype, device=target.device)

        loss = alpha * (1 - input) ** gamma * (input - target) ** 2
        if self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "mean":
            loss = torch.mean(loss)
        else:
            raise ("reduction must in [sum,mean]")
        return loss


class History():
    epoch = []
    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}

    # 打印训练结果信息
    def show_final_history(self,history):
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].set_title('loss')
        ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
        ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
        ax[1].set_title('acc')
        ax[1].plot(history.epoch, history.history["acc"], label="Train acc")
        ax[1].plot(history.epoch, history.history["val_acc"], label="Validation acc")
        ax[0].legend()
        ax[1].legend()

        plt.show()

class ClassifyModel(nn.Module):
    def __init__(self):
        super(ClassifyModel,self).__init__()
        # 每个类别对应的样本数
        nums_example=[1035,273,828,215,378,226,950,278,288,360,691,464,232,1212,2119,223]
        # 转成类别权重，样本数越少权重越大 使用 e^(-x)
        nums_example = np.asarray(nums_example)
        nums_example = (nums_example-np.min(nums_example))/(np.max(nums_example)-np.min(nums_example))
        nums_example = np.exp(-1*nums_example)
        self.weights = nums_example/np.sum(nums_example)*100

        self.num_classes = 16 # 10
        self.epochs = 10
        self.droprate = 0.3
        self.batch_size = 32
        self.test_batch_size = 64
        seed = 100
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        torch.manual_seed(seed)
        kwargs = {'num_workers': 5, 'pin_memory': True} if self.use_cuda else {}

        base_dir = "./datas/tomato2"
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_transformations = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomChoice([
                transforms.Resize((224,224)),
                transforms.RandomResizedCrop((224,224)),
                transforms.RandomCrop((224,224)),
                transforms.CenterCrop((224,224))
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
            transforms.Resize((256,256)),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),  # 转成0.～1.
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            normalize
        ])

        self.train_loader = DataLoader(
            MyData(base_dir, "train", transform=train_transformations),
            batch_size=self.batch_size, shuffle=True, **kwargs
        )

        # test_loader = DataLoader(
        #     MyData(base_dir, "valid", transform=test_transformations),
        #     batch_size=batch_size, shuffle=True, **kwargs
        # )

        self.test_loader = DataLoader(
            MyData(base_dir, "valid", transform=test_transformations),
            batch_size=self.test_batch_size, shuffle=True  # 加上**kwargs导致tensorrt加载数据失败
        )

        # load model
        # self.network = MyResnet(self.num_classes, "resnet18", 512, True, self.droprate)
        self.network = MyResnet(self.num_classes, "resnet50", 2048, True, self.droprate)
        # self.network = MyMnasnet(self.num_classes, "mnasnet1_0", 1280, True, self.droprate)
        # self.network = MyDensenet(self.num_classes, "densenet121", 1024, True, self.droprate)

        if self.use_cuda:
            self.network.to(self.device)
            # model = nn.DataParallel(model)

        # optimizer
        base_params = list(
            map(id, self.network.backbone.parameters())
        )
        logits_params = filter(lambda p: id(p) not in base_params, self.network.parameters())

        params = [
            {"params": logits_params, "lr": 1e-3},
            {"params": self.network.backbone.parameters(), "lr": 1e-4},
        ]
        self.optimizer = torch.optim.Adam(params, weight_decay=4e-05)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)

        self.history = History()

        self.save_model = "./model.pt"
        if os.path.exists(self.save_model):
            # model.load_state_dict(torch.load(self.save_model))
            state_dict = torch.load(self.save_model)
            self.network.load_state_dict({k: v for k, v in state_dict.items() if k in self.network.state_dict()})

        # 自定义loss
        self.lossFunc = LossFunc(self.num_classes,"sum")

    def forward(self):
        for epoch in range(self.epochs):
            train_acc, train_loss = self.__train(epoch)
            test_acc, test_loss = self.__test(epoch)

            # update the learning rate
            self.lr_scheduler.step()
            # save model
            torch.save(self.network.state_dict(), self.save_model)

            # 记录每个epoch的loss与acc
            self.history.epoch.append(epoch)
            self.history.history["loss"].append(train_loss)
            self.history.history["acc"].append(train_acc)
            self.history.history["val_loss"].append(test_loss)
            self.history.history["val_acc"].append(test_acc)

        # 保存json文件
        json.dump(self.history.history, open("result.json", "w"))

        # 打印训练结果
        self.history.show_final_history(self.history)

    def __train(self,epoch):
        self.network.train()
        train_loss = 0
        correct = 0
        num_trains = len(self.train_loader.dataset)
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.use_cuda:
                data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.network(data)
            # loss = F.cross_entropy(output, target, reduction="sum")
            loss = self.lossFunc.focal_cross_entropy(output,target,self.weights)
            # loss = self.lossFunc.focal_mse(output,target,self.weights)

            train_loss += loss.item()
            loss /= len(data)
            loss.backward()
            self.optimizer.step()

            # Compute accuracy
            # _, argmax = torch.max(output, 1)
            # accuracy = (target == argmax.squeeze()).float().mean()

            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            if batch_idx % 50 == 0:
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(
                #     epoch, batch_idx * len(data), len(train_loader),
                #            100. * batch_idx / len(train_loader), loss.item(), accuracy.item()))
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), num_trains,
                           100. * batch_idx * len(data) / num_trains, loss.item()))

        train_loss /= num_trains
        train_acc = correct / num_trains

        print('Train, Average Loss: {:.6f}\t,acc:{:.6f}'.format(
            train_loss, train_acc))

        return train_acc, train_loss

    def __test(self,epoch):
        self.network.eval()
        test_loss = 0
        correct = 0
        num_tests = len(self.test_loader.dataset)
        with torch.no_grad():
            for data, target in self.test_loader:
                if self.use_cuda:
                    data, target = data.to(self.device), target.to(self.device)
                output = self.network(data)
                # test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                test_loss += F.cross_entropy(output, target, reduction="sum").item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= num_tests
        test_acc = correct / num_tests
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, num_tests, 100 * test_acc))

        return test_acc, test_loss

if __name__=="__main__":
    model = ClassifyModel()
    model()
    """
    x = torch.rand(2, 3, 224, 224)
    # network = MyMnasnet(10, "mnasnet1_0", 1280, False, 0.5)
    network = MyDensenet(10, "densenet121", 1024, False, 0.5)
    y = network(x)
    print(y.shape)
    """