import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import trunc_normal_


class Decoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage

    Args:
        in_channels: input channels

    """
    def __init__(self, 
            in_channels=512, 
            channels=10, 
            num_classes=2, 
            dropout_ratio=0.1, 
            in_index=-1, 
            input_transform=None, 
            img_size=(512, 128), 
            embed_dim=512, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            num_conv=4, 
            align_corners=False,
            upsampling_method='bilinear', 
            num_upsampe_layer=4):
        super().__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.num_conv = num_conv
        self.norm = norm_layer(embed_dim)
        self.upsampling_method = upsampling_method
        self.num_upsampe_layer = num_upsampe_layer
        self.align_corners = align_corners
        out_channel=self.num_classes

        self.conv_0 = nn.Conv2d(embed_dim, 256, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(256, out_channel, kernel_size=1, stride=1)

        self.syncbn_fc_0 = nn.BatchNorm2d(256) 
        self.syncbn_fc_1 = nn.BatchNorm2d(256) 
        self.syncbn_fc_2 = nn.BatchNorm2d(256) 
        self.syncbn_fc_3 = nn.BatchNorm2d(256) 

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map will be selected. So in_channels and in_index must be of type int.

        Args:
            in_channels (int|Sequence[int]): Input channels.

            in_index (int|Sequence[int]): Input feature index.

            input_transform (str|None): Transformation type of input features.  
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the same size as first one and than concat together. Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']

        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, x):
        x = self._transform_inputs(x)
        #print(x.shape)

        if x.dim()==3:
            if x.shape[1] % 48 !=0:
                x = x[:,1:]
            x = self.norm(x)
        #print(x.shape)

        if self.upsampling_method=='bilinear':
            if x.dim()==3:
                n, hw, c = x.shape
                h = w = int(math.sqrt(hw))
                x = x.transpose(1,2).reshape(n, c, h, w)

            if self.num_conv==2:
                if self.num_upsampe_layer==2:
                    x = self.conv_0(x)
                    x = self.syncbn_fc_0(x)
                    x = F.relu(x,inplace=True)
                    x = F.interpolate(x, size=x.shape[-1]*4, mode='bilinear', align_corners=self.align_corners)
                    x = self.conv_1(x)
                    x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=self.align_corners)
                elif self.num_upsampe_layer==1:
                    x = self.conv_0(x)
                    x = self.syncbn_fc_0(x)
                    x = F.relu(x,inplace=True)
                    x = self.conv_1(x)
                    x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=self.align_corners)
            elif self.num_conv==4:
                if self.num_upsampe_layer==4:
                    x = self.conv_0(x)
                    x = self.syncbn_fc_0(x)
                    x = F.relu(x,inplace=True)
                    x = F.interpolate(x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
                    x = self.conv_1(x)
                    x = self.syncbn_fc_1(x)
                    x = F.relu(x,inplace=True)
                    x = F.interpolate(x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
                    x = self.conv_2(x)
                    x = self.syncbn_fc_2(x)
                    x = F.relu(x,inplace=True)
                    x = F.interpolate(x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
                    x = self.conv_3(x)
                    x = self.syncbn_fc_3(x)
                    x = F.relu(x,inplace=True)
                    x = self.conv_4(x)
                    #x = F.interpolate(x, size=x.shape[-1]*2, mode='bilinear', align_corners=self.align_corners)
                    x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=self.align_corners)
        return x
