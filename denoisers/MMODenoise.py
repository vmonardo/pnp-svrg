#!/usr/bin/env python
# coding=utf-8
try:
    from .denoiser import Denoise
    from .cnn.cnn import Denoiser, DnCNN
except:
    from denoiser import Denoise
    from cnn.cnn import Denoiser, DnCNN
import torch
import torch.nn as nn
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def apply_model(x_cur, model=None):

    imgn = torch.from_numpy(x_cur)
    init_shape = imgn.shape
    if len(init_shape) == 2:  
        imgn.unsqueeze_(0)
        imgn.unsqueeze_(0)
    elif len(init_shape) == 3:
        imgn.unsqueeze_(0)
    imgn = imgn.type(Tensor)                          

    with torch.no_grad():
        imgn.clamp_(0, 1)  # Might be necessary for some networks
        out_net = model(imgn)
        out_net.clamp_(0, 1)

    img = out_net[0, ...].cpu().detach().numpy()
    if len(init_shape) == 2:
        x = img[0]     
    elif len(init_shape) == 3:
        x = img                 

    return x 

def load_net(pth=None, net_type='DnCNN_nobn', channels=1, n_lev=0.01, cuda=True, root_folder='.'):

    if 'DnCNN_nobn' in net_type:
        avg, bn, depth = False, False, 20
        net = simple_CNN(n_ch_in=channels, n_ch_out=channels, n_ch=64, nl_type='relu', depth=depth, bn=bn)
        pth = root_folder+'checkpoints/pretrained/'+net_type+'_nch_'+str(channels)+'_nlev_'+str(n_lev)+'.pth'
        
    if cuda:
        cuda_infotxt = "cuda driver found - moving to GPU.\n"
        print(cuda_infotxt)
        net.cuda()

    if cuda:
        model = nn.DataParallel(net).cuda()
    else:
        model = nn.DataParallel(net)
            
    if pth is not None:
        loaded_txt = "Loading " + pth + "...\n"
        print(loaded_txt)
        model = load_checkpoint(model, pth)
    else:
        raise NameError('Could not load '+str(net_type))
    
    return model.eval() 

def load_checkpoint(model, filename):
    checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    model.module.load_state_dict(checkpoint.module.state_dict())
    return model

class simple_CNN(nn.Module):
    def __init__(self, n_ch_in=3, n_ch_out=3, n_ch=64, nl_type='relu', depth=5, bn=False):
        super(simple_CNN, self).__init__()

        self.nl_type = nl_type
        self.depth = depth
        self.bn = bn

        self.in_conv = nn.Conv2d(n_ch_in, n_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_list = nn.ModuleList([nn.Conv2d(n_ch, n_ch, kernel_size=3, stride=1, padding=1, bias=True) for _ in range(self.depth-2)])
        self.out_conv = nn.Conv2d(n_ch, n_ch_out, kernel_size=3, stride=1, padding=1, bias=True)

        if self.nl_type == 'relu':
            self.nl_list = nn.ModuleList([nn.LeakyReLU() for _ in range(self.depth-1)])
        if self.bn:
            self.bn_list = nn.ModuleList([nn.BatchNorm2d(n_ch) for _ in range(self.depth-2)])

    def forward(self, x_in):

        x = self.in_conv(x_in)
        x = self.nl_list[0](x)

        for i in range(self.depth-2):
            x_l = self.conv_list[i](x)  
            if self.bn:
                x_l = self.bn_list[i](x_l)
            x = self.nl_list[i+1](x_l)
        
        x_out = self.out_conv(x)+x_in  # Residual skip

        return x_out

class MMODenoiser(Denoise):
    def __init__(   self, model=None, 
                    channels=3, 
                    path=None, 
                    cuda=True, 
                    sigma=0.01, 
                    root_path='.'):
        super().__init__()

        """ 
        Intialise the denoiser
        """
        self.sigma = sigma

        if model is None:
            self.network = load_net(path, net_type='DnCNN_nobn', channels=channels, cuda=cuda, root_folder=root_path, n_lev=self.sigma)       # Path of the saved model in the case the denoiser we use is a network
        else:
            self.network = model

    def denoise(self, noisy):
        self.t += 1
        noisy = np.moveaxis(noisy, -1, 0)
        tmp = apply_model(noisy, model=self.network)
        return np.clip(np.moveaxis(tmp, 0, -1), 0., 1.)

if __name__ == '__main__':
    architecture='DnCNN_nobn' 
    n_ch=3
    n_lev=0.01 
    noise_level_den=0.007
    H, W = 128, 128
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    torch.backends.cudnn.benchmark = True

    denoiser = MMODenoiser(channels=n_ch, cuda=cuda, sigma=noise_level_den, root_path='./')

    # image_path = './BSDS300/images/test/3096.jpg'
    # image_path='../data/Set12/01.png'
    image_path = '../data/RGB/8023.jpg'
    image_true = Image.open(image_path).resize((H, W))
    image_true = np.asarray(image_true, dtype="float32")/255.

    noise = np.random.randn(*image_true.shape)
    y = image_true+n_lev*noise
    y_denoise = denoiser.denoise(y)

    y0 = np.clip(y, 0., 1.)
    y1 = y_denoise

    f, axarr = plt.subplots(2,1)
    axarr[0].imshow(y0, cmap='gray', vmin=0, vmax=1)
    axarr[1].imshow(y1, cmap='gray', vmin=0, vmax=1)

    plt.show()
