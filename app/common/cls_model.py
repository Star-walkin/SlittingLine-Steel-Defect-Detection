import os
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, wide_resnet50_2
try:
    # torchvision>=0.13
    from torchvision.models import ResNet18_Weights
except Exception:
    ResNet18_Weights = None  # type: ignore


class ProjectionNet(nn.Module):
    def __init__(self, pretrained=True, head_layers=[512,512,512,512,512,512,512,512,128], num_classes=2):
        super(ProjectionNet, self).__init__()
        #self.resnet18 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=pretrained)
        #self.resnet18 = resnet18(pretrained=pretrained)
        # 兼容 torchvision 新版 API：pretrained 已弃用，使用 weights
        if ResNet18_Weights is not None:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.resnet18 = resnet18(weights=weights)
        else:
            # 旧版本 fallback
            self.resnet18 = resnet18(pretrained=pretrained)
        # create MPL head as seen in the code in: https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        # TODO: check if this is really the right architecture
        last_layer = 512
        #last_layer = 2048
        sequential_layers = []
        for num_neurons in head_layers:
            sequential_layers.append(nn.Linear(last_layer, num_neurons))
            sequential_layers.append(nn.BatchNorm1d(num_neurons))
            sequential_layers.append(nn.ReLU(inplace=True))
            last_layer = num_neurons
        
        #the last layer without activation

        head = nn.Sequential(
            *sequential_layers
          )
        self.resnet18.fc = nn.Identity()#f(x)=x,恒等函数
        self.head = head
        self.out = nn.Linear(last_layer, num_classes)

    
    def forward(self, x):
        embeds = self.resnet18(x)
        tmp = self.head(embeds)
        logits = self.out(tmp)

        return  logits
    
    def freeze_resnet(self):
        # freez full resnet18
        for param in self.resnet18.parameters():
            param.requires_grad = False
        
        #unfreeze head:
        for param in self.resnet18.fc.parameters():
            param.requires_grad = True


    def unfreeze(self):
        #unfreeze all:
        for param in self.parameters():
            param.requires_grad = True
        print(param.requires_grad)
#print(ProjectionNet())