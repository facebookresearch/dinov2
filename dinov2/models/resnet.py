import torch
import torch.nn as nn
import torchvision.models as models

class CustomResNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Load pretrained ResNet50 model
        resnet = models.resnet50(pretrained=pretrained)
        
        # Split model into layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        layer4_output = self.layer4(x)
        
        pooled = self.avgpool(layer4_output)
        embeddings = torch.flatten(pooled, 1)
        
        return {
            'layer4_output': layer4_output,
            'embeddings': embeddings
        }
