from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import numpy as np
import os
# from torch.utils.data import DataLoader, Dataset
# from torchvision import datasets, transforms
# import PIL.Image
from collections import OrderedDict
import math
import matplotlib.pyplot as plt

from data_prepare import train_loader,test_loader

class BottleNet(nn.Module):
    def __init__(self,cfg1=[],cfg2=[]):
        super(BottleNet, self).__init__()
        self.conv1=nn.Conv2d(cfg1[0],cfg1[1],cfg1[2],stride=cfg1[3],padding=cfg1[4])
        self.bn1=nn.BatchNorm2d(cfg1[1])
        self.relu1=nn.ReLU()

        self.conv2=nn.Conv2d(cfg2[0],cfg2[1],cfg2[2],stride=cfg2[3],padding=cfg2[4])
        self.bn2 = nn.BatchNorm2d(cfg2[1])
        self.relu2 = nn.ReLU()

        self.stride=cfg1[3]
        if self.stride==2:
            self.downsample=nn.Sequential(
                nn.Conv2d(cfg1[0],cfg2[1],1,stride=2,padding=0),
                nn.BatchNorm2d(cfg2[1])
            )

    def forward(self,x):
        identity = x
        net=self.conv1(x)
        net=self.bn1(net)
        net=self.relu1(net)

        net=self.conv2(net)
        net=self.bn2(net)

        if self.stride==2:
            identity=self.downsample(x)

        net=self.relu2(identity+net)

        return net

# 参考resnet18自定义模型
class MyModel(nn.Module):
    def __init__(self, num_classes=10, droprate=0.0):
        super(MyModel, self).__init__()
        self.layer0=nn.Sequential(OrderedDict([
            ("conv1",nn.Conv2d(3,64,7,stride=2,padding=3)),
            ("bn1",nn.BatchNorm2d(64)),
            ("relu1",nn.ReLU()),
            ('maxpool1',nn.MaxPool2d(3,stride=2,padding=1))]))


        self.layer1_0=BottleNet([64,64,3,1,1],[64,64,3,1,1])
        self.layer1_1=BottleNet([64,64,3,1,1],[64,64,3,1,1])

        self.layer2_0=BottleNet([64,128,3,2,1],[128,128,3,1,1])
        self.layer2_1=BottleNet([128,128,3,1,1],[128,128,3,1,1])

        self.layer3_0 = BottleNet([128, 256, 3, 2, 1], [256, 256, 3, 1, 1])
        self.layer3_1 = BottleNet([256, 256, 3, 1, 1], [256, 256, 3, 1, 1])

        self.layer4_0 = BottleNet([256, 512, 3, 2, 1], [512, 512, 3, 1, 1])
        self.layer4_1 = BottleNet([512, 512, 3, 1, 1], [512, 512, 3, 1, 1])

        # self.layer5=nn.Sequential(
        #     nn.Dropout(droprate),
        #     nn.Conv2d(512, num_classes, 1, stride=1,padding=0),
        #     nn.BatchNorm2d(num_classes),
        #     nn.ReLU(),
        #     nn.AvgPool2d(1,stride=7,padding=0)
        # )
        self.layer5 = nn.Sequential(OrderedDict([
            ("dropout",nn.Dropout(droprate)),
            ("conv1",nn.Conv2d(512, num_classes, 1, stride=1,padding=0)),
            ("bn1", nn.BatchNorm2d(num_classes)),
            ("relu1", nn.ReLU()),
            ('avgpool1', nn.AvgPool2d(1,stride=7,padding=0))]))

    def forward(self,x):
        x=self.layer0(x)
        x=self.layer1_0(x)
        x=self.layer1_1(x)

        x = self.layer2_0(x)
        x = self.layer2_1(x)

        x = self.layer3_0(x)
        x = self.layer3_1(x)

        x = self.layer4_0(x)
        x = self.layer4_1(x)

        x = self.layer5(x)

        x = torch.flatten(x, 1)

        return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    num_trains = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target, reduction="sum")
        train_loss += loss.item()
        loss /= len(data)
        loss.backward()
        optimizer.step()

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


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    num_tests = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= num_tests
    test_acc = correct / num_tests
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, num_tests, 100 * test_acc))

    return test_acc, test_loss


# 打印训练结果信息
def show_final_history(history):
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


if __name__ == "__main__":
    num_classes = 10
    # batch_size = 64
    epochs = 2
    droprate = 0.3
    # seed = 100
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # torch.manual_seed(seed)
    # kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}

    # base_dir = os.path.join(os.getcwd(), "./tomato")
    # base_dir = "/media/wucong/work/practice/data/tomato"
    # train_dir =os.path.join(base_dir,"train")
    # val_dir =os.path.join(base_dir,"valid")


    # model = MyModel2(num_classes, "resnet18", 512, True, droprate)
    model = MyModel(num_classes, droprate)
    model.to(device)
    # model.train()
    # model = nn.DataParallel(model)

    # optimizer
    """
    base_params = list(
        map(id, model.backbone.parameters())
    )
    logits_params = filter(lambda p: id(p) not in base_params, model.parameters())

    params = [
        {"params": logits_params, "lr": 1e-3},
        {"params": model.backbone.parameters(), "lr": 1e-4},
    ]
    """
    params=model.parameters()
    optimizer = torch.optim.Adam(params, weight_decay=4e-05)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    valid_loss = math.inf


    class History():
        epoch = []
        history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}


    history = History()
    
    if not os.path.exists("model.pt"):
        for epoch in range(epochs):
            train_acc, train_loss = train(model, device, train_loader, optimizer, epoch)

            # 根据验证loss保存模型参数
            test_acc, test_loss = test(model, device, test_loader)
            if test_loss < valid_loss:
                torch.save(model.state_dict(), "model.pt")
                valid_loss = test_loss
                print("save model success!")

            # update the learning rate
            lr_scheduler.step()
            # torch.save(model.state_dict(), "model.pt")

            history.epoch.append(epoch)
            history.history["loss"].append(train_loss)
            history.history["acc"].append(train_acc)
            history.history["val_loss"].append(test_loss)
            history.history["val_acc"].append(test_acc)

        show_final_history(history)

    else:
        # model.load_state_dict(torch.load("model.pt"))
        state_dict = torch.load("./model.pt")
        model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict()})

        # save to npz
        weights_arg = {}
        for key, value in model.state_dict().items():
            weights_arg[key] = value.cpu().numpy()

        np.savez("model.npz", **weights_arg)


        # save onnx
        batch_size=32
        dummy_input = torch.randn(batch_size, 3, 224, 224).to("cuda")
        torch.onnx.export(model, dummy_input, "model.onnx", verbose=True, export_params=True)
