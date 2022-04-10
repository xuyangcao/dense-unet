import torch 
from denseunet import DenseUnet
from resunet import ResUNet
from unet import UNet

#model = DenseUnet(arch='121')
#model = ResUNet(3, 2, False)
model = UNet(3, 2)
n_params = sum([p.data.nelement() for p in model.parameters()])
print(n_params)
