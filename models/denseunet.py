import torch 
import torch.nn as nn
from torchvision.models import densenet
import torch.nn.functional as F
from collections import OrderedDict
import math

class UpSampeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampeBlock, self).__init__()
        self.up = nn.functional.interpolate
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        # init weights 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, out=None):
        x = self.up(x, scale_factor=2, mode='bilinear', align_corners=False)
        if out is not None:
            x = torch.cat([x, out], 1)
        x = self.relu2(self.bn2(self.conv(x))) 
        return x

class OutputBlock(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(OutputBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=False)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv(x)
        return out 

class DenseUnet(nn.Module):
    def __init__(self, arch='121', pretrained=True, num_classes=2, drop_rate=0.3):
        super(DenseUnet, self).__init__()
        if arch == '121':
           features = densenet.densenet121(pretrained=pretrained, drop_rate=drop_rate).features
        if arch == '161': 
            features = densenet.densenet161(pretrained=pretrained, drop_rate=drop_rate).features
        if arch == '201':
            features = densenet.densenet201(pretrained=pretrained, drop_rate=drop_rate).features
            
        # encorder
        self.input_block = features[:4]
        self.dense_block_1 = features[4]
        self.transition_block_1 = features[5]
        self.dense_block_2 = features[6]
        self.transition_block_2 = features[7]
        self.dense_block_3 = features[8]
        self.transition_block_3 = features[9]
        self.dense_block_4 = features[10]
        self.transition_block_4 = nn.Sequential(OrderedDict([('norm5', features[11]), ('relu', nn.ReLU(inplace=True))]))

        # decorder
        if arch == '121':
            self.up_1 = UpSampeBlock(1024+1024, 512)
            self.up_2 = UpSampeBlock(512+512, 256)
            self.up_3 = UpSampeBlock(256+256, 96)
        if arch == '161':
            self.up_1 = UpSampeBlock(2208+2112, 768)
            self.up_2 = UpSampeBlock(768+768, 384)
            self.up_3 = UpSampeBlock(384+384, 96)
        if arch == '201':
            self.up_1 = UpSampeBlock(1920+1792, 512)
            self.up_2 = UpSampeBlock(512+512, 256)
            self.up_3 = UpSampeBlock(256+256, 96)
        self.up_4 = UpSampeBlock(96, 96)
        self.up_5 = UpSampeBlock(96, 64)
        self.output_block = OutputBlock(64, num_classes)

    def forward(self, x):
        x_64 = self.input_block(x)
        x_64 = self.dense_block_1(x_64)
        x_32 = self.transition_block_1(x_64)
        x_32 = self.dense_block_2(x_32)
        x_16 = self.transition_block_2(x_32)
        x_16 = self.dense_block_3(x_16)
        x_8 = self.transition_block_3(x_16)
        x_8 = self.dense_block_4(x_8)
        out = self.transition_block_4(x_8)
        #print('out.shape: {}, x16.shape: {}'.format(out.shape, x_16.shape))
        out = self.up_1(out, x_16)
        #print('out.shape: {}, x32.shape: {}'.format(out.shape, x_32.shape))
        out = self.up_2(out, x_32)
        #print('out.shape: {}, x64.shape: {}'.format(out.shape, x_64.shape))
        out = self.up_3(out, x_64)
        #print('out.shape: {}'.format(out.shape))
        out = self.up_4(out)
        out = self.up_5(out)
        out = self.output_block(out)
        return out
