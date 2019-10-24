import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def passthrough(x, **kwargs):
    return x

def RELUCons(relu, nchan):
    if relu:
        return nn.RELU(inplace=True)
    else:
        return nn.PReLU(nchan)


class LUConv(nn.Module): 
    def __init__(self, nchan, relu):
        super(LUConv, self).__init__()
        self.relu1 = RELUCons(relu, nchan)
        self.conv1 = nn.Conv2d(nchan, nchan, kernel_size=3, padding=1)
        #self.conv1 = nn.Conv2d(nchan, nchan, kernel_size=5, padding=2)
        #self.bn1 = ContBatchNorm2d(nchan)
        self.bn1 = nn.BatchNorm2d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, relu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, relu))
    return nn.Sequential(*layers)

class InputTransition(nn.Module):
    def __init__(self, in_channels, out_channels, relu):
        super(InputTransition, self).__init__()
        self.out_channels = out_channels
        #self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        #self.bn1 = ContBatchNorm2d(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = RELUCons(relu, out_channels)
        self.relu2 = RELUCons(relu, out_channels)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))

        #x_64 = x
        #for _ in range(self.out_channels):
        #    x_64 = torch.cat((x_64, x), dim=1)
            
        #print("x.shape: ", x.shape) 
        x32 = torch.cat((x, x, x,
            x, x, x, x, x, x, x, x[:, :2, :, :]), 1)

        #print(out.shape)
        #print(x32.shape)
        out = self.relu2(torch.add(out, x32))
        return out

class DownTransition(nn.Module):
    def __init__(self, in_channels, n_convs, relu, dropout=False):
        super(DownTransition, self).__init__()
        out_channels = 2 * in_channels
        # sample by convolution
        self.down_conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        #self.bn1 = ContBatchNorm2d(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = RELUCons(relu, out_channels)
        self.relu2 = RELUCons(relu, out_channels)
        self.ops = _make_nConv(out_channels, n_convs, relu)
        self.do1 = passthrough
        if dropout:
            self.do1 = nn.Dropout2d()
    
    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        down = self.do1(down)
        out = self.ops(down)
        #print('down:', out.shape)
        #print(down.shape)
        out = self.relu2(torch.add(out, down))
        return out

class UpTransition(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs, relu, dropout=False):
        super(UpTransition, self).__init__()
        #print('{} / {}'.format(in_channels, out_channels))
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels // 2, kernel_size=2, stride=2)
        #self.bn1 = ContBatchNorm2d(out_channels // 2)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.relu1 = RELUCons(relu, out_channels // 2)
        self.relu2 = RELUCons(relu, out_channels)
        self.ops = _make_nConv(out_channels, n_convs, relu)
        self.do1 = passthrough
        if dropout:
            self.do1 = nn.Dropout2d()

    def forward(self, x, skipx):
        x = self.do1(x)
        out = self.relu1(self.bn1(self.up_conv(x)))
        x_cat = torch.cat((out, skipx), dim=1)
        out = self.ops(x_cat)
        out = self.relu2(torch.add(out, x_cat)) 
        return out  

class OutputTransition(nn.Module):
    def __init__(self, in_channels, n_classes, relu):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, n_classes, kernel_size=5, padding=2)
        #self.conv1 = nn.Conv2d(in_channels, n_classes, kernel_size=1, padding=0)
        #self.bn1 = ContBatchNorm2d(n_classes)
        self.bn1 = nn.BatchNorm2d(n_classes)
        self.relu1 = RELUCons(relu, n_classes)
        self.conv2 = nn.Conv2d(n_classes, n_classes, kernel_size=1)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x))) 
        out = self.conv2(out)
        
        #out = torch.softmax(out, dim=1)
        return out


class ResUNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=2, relu=False):
        super(ResUNet, self).__init__()

        self.input_tr = InputTransition(3, 32, relu)
        self.down_tr64 = DownTransition(32, 5, relu, dropout=False)
        self.down_tr128 = DownTransition(64, 5, relu, dropout=False)
        self.down_tr256 = DownTransition(128, 5, relu, dropout=False)
        self.down_tr512 = DownTransition(256, 5, relu, dropout=False)
        
        self.up_tr512 = UpTransition(512, 512, 5, relu, dropout=False)
        self.up_tr128 = UpTransition(512, 256, 5, relu, dropout=False)
        self.up_tr64 = UpTransition(256, 128, 5, relu, dropout=False)
        self.up_tr32 = UpTransition(128, 64, 5, relu, dropout=False)

        self.out_tr = OutputTransition(64, num_classes, relu)

    def forward(self, x):
        out32 = self.input_tr(x)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out512 = self.down_tr512(out256)

        out_512 = self.up_tr512(out512, out256)
        out_256 = self.up_tr128(out_512, out128)
        out_128 = self.up_tr64(out_256, out64)
        out_64 = self.up_tr32(out_128, out32)

        out = self.out_tr(out_64)
        
        return out 
