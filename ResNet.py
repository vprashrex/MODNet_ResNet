import torch
from torch import nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torch import nn as nn



def conv3x3(inplanes,planes,stride=1):
    return nn.Conv2d(inplanes,planes,3,stride,1,bias=False)

def conv1x1(inplanes,planes):
    return nn.Conv2d(inplanes,planes,1,bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(BasicBlock,self).__init__()
        
        self.conv1 = conv3x3(inplanes,planes,stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(True)
        
        self.conv2 = conv3x3(planes,planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self,x):
        
        res = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            res = self.downsample(x)
        
        out += res
        out = self.relu(out)
        
        return out
    
class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(Bottleneck,self).__init__()
        
        self.conv1 = conv1x1(inplanes,planes)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = conv3x3(planes,planes,stride)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = conv1x1(planes,planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.relu = nn.ReLU(True)
        
        self.downsample = downsample
        
        self.stride = stride
    
    def forward(self,x):
        
        res = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            res = self.downsample(x)
        
        out += res
        out = self.relu(out)
        
        return out
    
class ResNet(nn.Module):
    def __init__(self,block,layers,num_classes=3):
        self.inplanes = 64
        
        super(ResNet,self).__init__()
        
        self.conv1 = nn.Conv2d(
            3,64,7,2,3,bias=False
        )
        
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3,2,1)
        
        self.layer1 = self._make_layer(block,64,layers[0])
        self.layer2 = self._make_layer(block,128,layers[1],stride=2)
        self.layer3 = self._make_layer(block,256,layers[2],stride=2)
        self.layer4 = self._make_layer(block,512,layers[3],stride=2)
        
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.weight,0)
                
    def _make_layer(self,block,planes,blocks,stride=1):
        
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,planes * block.expansion,1,stride,bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
            
        
        layers = []
        
        layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes = planes * block.expansion
        
        for i in range(1,blocks):
            layers.append(block(self.inplanes,planes))
        
        return nn.Sequential(*layers)

    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.maxpool(x)
        x = self.layer1(x)
        
        x = self.layer2(x)
        
        x = self.layer3(x)
        
        x = self.layer4(x)
        
        return x
    
    def _load_pretrained_ckpt(self,pretrained_dict):
        
        model_dict = {}
        state_dict = self.state_dict()
        
        print('[ResNet50]--Loading pretrained model...')
        
        for k,v in pretrained_dict.items():
            if k in state_dict:
                model_dict[k] = v
            
            else:
                print(k,' is ignored !')
            
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


class BaseBackbone(nn.Module):
    def __init__(self):
        super(BaseBackbone,self).__init__()
        
        self.model = None
    
    def forward(self,x):
        raise NotImplementedError
    
    def load_pretrained_ckpt(self):
        raise NotImplementedError
    

class ResNet50Backbone(BaseBackbone):
    def __init__(self):
        super(ResNet50Backbone,self).__init__()
        
        self.model = ResNet(Bottleneck,[3,4,6,3])

        self.enc_channels = [64,256,512,1024,2048]
        
        
        
    def forward(self,x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        
        enc2x = x
        
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        
        enc4x = x
        
        x = self.model.layer2(x)
        
        enc8x = x
        
        x = self.model.layer3(x)
        
        enc16x = x
        
        x = self.model.layer4(x)
        
        enc32x = x
        
        return [enc2x,enc4x,enc8x,enc16x,enc32x]
    
    def load_pretrained_ckpt(self):
        self.model._load_pretrained_ckpt(model_zoo.load_url(
            'https://download.pytorch.org/models/resnet50-19c8e357.pth'))


if __name__ == '__main__':
    input = torch.randn(1,3,512,512)
    backbone = ResNet50Backbone()
    backbone.load_pretrained_ckpt()

    enc2x, enc4x, enc8x, enc16x, enc32x = backbone(input)
    
    print(enc2x.shape)
    print(enc4x.shape)
    print(enc8x.shape)
    print(enc16x.shape)
    print(enc32x.shape)
