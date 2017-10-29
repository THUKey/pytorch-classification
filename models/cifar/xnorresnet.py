from __future__ import absolute_import

'''Resnet for cifar dataset. 
Ported form 
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei 
'''
import torch.nn as nn
import math
import torch
import torch.nn.functional as F


__all__ = ['xnorresnet']


class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        mean = torch.mean(input.abs(), 1, keepdim=True)
        input = input.sign()
        return input, mean

    def backward(self, grad_output, grad_output_mean):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class BinConv2d(nn.Module):
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, dropout=0):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(input_channels, output_channels,
                kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.bn(x)
        x, mean = BinActive()(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        x = self.conv(x)
        x = self.relu(x)
        return x

class BinConv2d_norelu(nn.Module):
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, dropout=0):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(input_channels, output_channels,
                kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        #self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.bn(x)
        x, mean = BinActive()(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        x = self.conv(x)
        #x = self.relu(x)
        return x

#def conv3x3(in_planes, out_planes, stride=1):
#    "3x3 convolution with padding"
#    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                     padding=1, bias=False)

def binconv3x3(in_planes, out_planes, stride=1):
    "3x3 binconvolution with padding"
    return BinConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1)

def binconv3x3_norelu(in_planes, out_planes, stride=1):
    "3x3 binconvolution with padding"
    return BinConv2d_norelu(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1)

class BinBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BinBasicBlock, self).__init__()
        self.conv1 = binconv3x3(inplanes, planes, stride)
        #self.bn1 = nn.BatchNorm2d(planes)
        #self.relu = nn.ReLU(inplace=True)
        self.conv2 = binconv3x3_norelu(planes, planes)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        #out = self.bn1(out)
        #out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        #TODO: relu ?
        out = self.relu(out)

        return out


class BinBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BinBottleneck, self).__init__()
        #self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv1 = BinConv2d(inplanes, planes, kernel_size=1, stride=1,
                               padding=1)
        #self.bn1 = nn.BatchNorm2d(planes)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
        #                       padding=1, bias=False)
        self.conv2 = BinConv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1)
        #self.bn2 = nn.BatchNorm2d(planes)
        #self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.conv3 = BinConv2d_norelu(planes, planes * 4, kernel_size=1, stride=1,
                               padding=1)
        #self.bn3 = nn.BatchNorm2d(planes * 4)
        #self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        #out = self.bn1(out)
        #out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)
        #out = self.relu(out)

        out = self.conv3(out)
        #out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # TODO: relu?
        out = self.relu(out)

        return out


class XnorResNet(nn.Module):

    def __init__(self, depth, num_classes=1000):
        super(XnorResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = BinBottleneck if depth >=44 else BinBasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = BinConv2d_norelu(self.inplanes, planes * block.expansion,
                                   kernel_size=1, stride=1, padding=1)
            #downsample = nn.Sequential(
            #    nn.Conv2d(self.inplanes, planes * block.expansion,
            #              kernel_size=1, stride=stride, bias=False),
            #    nn.BatchNorm2d(planes * block.expansion),
            #)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def xnorresnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return XnorResNet(**kwargs)
