"""
pytorch python api加载 C++ api训练的参数

https://github.com/pytorch/pytorch/issues/21679
BatchNorm: C++ Frontend ignoring running_mean and running_var
加载C++ 权重中没有running_mean，running_var 导致python api加载失败

解决方法：
1.python api要调用C++ api 可以将running_mean设置为0,running_var设置为1  精度很低
2.C++ api中不使用BatchNorm， 训练的精度不高
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
import numpy as np

class MyModel(nn.Module):
    def __init__(self, num_classes=10,p=0.2):
        super(MyModel, self).__init__()
        self.conv1=nn.Conv2d(1,10,3,stride=1,padding=1,bias=False)
        self.maxpool1=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(10,20,3,stride=1,padding=1,bias=False)
        self.maxpool2 = nn.MaxPool2d(2)
        self.batch_norm=nn.BatchNorm2d(20)
        self.fc1=nn.Linear(980,50)
        self.dropout=nn.Dropout(p)
        self.fc2=nn.Linear(50,num_classes)

    def forward(self, x):
        x=self.conv1(x)
        x=self.maxpool1(x)
        x=F.relu(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = F.relu(x)
        x = self.batch_norm(x)

        x=x.view(-1,980)
        x=F.relu(self.fc1(x))
        x=self.dropout(x)
        x = F.log_softmax(self.fc2(x))

        return x

criterion=nn.CrossEntropyLoss(reduction="mean")

def accuracy2(output,target):
    _, argmax = torch.max(output, 1)
    accuracy = (target == argmax.squeeze()).float().mean()
    return accuracy

def test(model, device, test_loader):
    model.eval()
    # test_loss = 0
    # correct = 0
    avg_loss = []
    avg_acc = []
    # num_tests=len(test_loader.dataset)
    with torch.no_grad():
        for data,target in test_loader:
            data=data.to(device)
            target=target.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            # test_loss += F.cross_entropy(output, target,reduction="sum").item() # sum up batch loss
            # pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()
            # loss = criterion(output, target)
            loss = criterion(output, target)
            # acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # acc = accuracy(output, target)[0]
            acc = accuracy2(output, target)


            avg_loss.append(loss.item())
            avg_acc.append(acc.item())

    # test_loss /= num_tests
    # test_acc=correct / num_tests

    test_loss = np.mean(avg_loss)
    test_acc = np.mean(avg_acc)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
    #     test_loss, correct, num_tests,100*test_acc))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.4f}\n'.format(
        test_loss, test_acc))

    return test_acc,test_loss


if __name__=="__main__":
    torch.manual_seed(100)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # 从C++ AP 加载权重
    model=MyModel()
    model.to(device)

    # torch.save(model.state_dict(), "model.pt")
    # torch.save(model, "model-checkpoint.pt")
    # exit(0)

    # model.load_state_dict(torch.load("./model.pt"))
    module = torch.jit.load("./build/model-checkpoint.pt")
    model.load_state_dict({k:v for k,v in dict(module.named_parameters()).items() if k in model.state_dict()}) # list(module.parameters())
    """
    parms_dict={}
    ori_parms=dict(module.named_parameters())
    for k, v in model.state_dict().items():
        if k in ori_parms:
            parms_dict[k]=ori_parms[k]
        else:
            if "running_mean" in k:
                parms_dict[k]=torch.zeros_like(v)
            if "running_var" in k:
                parms_dict[k] = torch.ones_like(v)

    model.load_state_dict(parms_dict) # list(module.parameters())
    """
    test_loader = DataLoader(
        datasets.MNIST("/media/wucong/work/practice/work/data", train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), 64, shuffle=False, num_workers=1
    )

    test(model,device,test_loader)
