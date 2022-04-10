import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder

class EncoderDecoder(nn.Module):
    def __init__(self, 
            pretrained=None, 
            **kwargs
            ): 
        super().__init__()
        self.backbone = Encoder(**kwargs) 
        self.decode_head = Decoder(**kwargs)

        self._init_weights(pretrained=pretrained)

    def _init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()

    def forward(self, x):
        x = self.backbone(x)
        x = self.decode_head(x)
        return x

segmentor = EncoderDecoder()
x = torch.randn((1, 3, 512, 128))
segmentor = segmentor.to('cuda:0')
x = x.to('cuda:0')
print(x.shape)
out = segmentor(x)
print(out.shape)
