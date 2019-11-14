from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import numpy as np
import os
import shutil
from collections import OrderedDict
from tqdm import tqdm

# model
class MyModel(nn.Module):
    def __init__(self, num_classes, model_name="resnet101", backbone_size=2048, pretrained=False, droprate=0.0,
                 device="cpu"):
        super(MyModel, self).__init__()
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


def predict(model, device, test_loader,category=[]):
    model.eval()
    with torch.no_grad():
        for data, path in tqdm(test_loader):
            data = data.to(device)
            output = F.softmax(model(data),-1)
            score,pred = output.max(1, keepdim=False)
            for _score,_pred,_path in zip(score,pred,path):
                label=category[int(_pred)]
                if _score<0.8:label="unknow"
                save_path=os.path.join(os.path.dirname(_path),label)
                if not os.path.exists(save_path):os.makedirs(save_path)
                # save_path=os.path.join(save_path,os.path.basename(_path).replace(".jpg","_%0.3f.jpg"%(_score)))
                save_path=os.path.join(save_path,os.path.basename(_path))
                shutil.move(_path,save_path)

# base_dir = "/media/wucong/work/practice/data/tomato"
# category=sorted(os.listdir(os.path.join(base_dir, "train")))
# print(category)
category=['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
          'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two_spotted_spider_mite', 'Tomato___Target_Spot',
          'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
