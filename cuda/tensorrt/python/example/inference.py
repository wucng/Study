from data_prepare import test_loader
from model import MyModel,test
import torch

num_classes = 10
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = MyModel(num_classes, "resnet18", 512, False, 0.0)
model.to(device)
state_dict = torch.load("./model.pt")
model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict()})

test_acc, test_loss = test(model, device, test_loader)

"""Test set: Average loss: 0.1198, Accuracy: 4520/4585 (99%)"""