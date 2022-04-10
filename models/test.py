#from unet_1 import UNet
#import torch 
#
#x = torch.randn((10, 3, 256, 256))
#
#model = UNet(3, 2)
#print(model)
#
#y = model(x)
#print('x.shape: ', x.shape)
#print('y.shape: ', y.shape)


class Test():
    def __call__(self, x):
        print('call method')
        self.forward(x)
    
    def forward(self, x):
        print('forward method')
        print(x)

t = Test()
t(100)
print('-----------')
t.forward(100)
