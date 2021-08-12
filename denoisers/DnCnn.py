#!/usr/bin/env python
# coding=utf-8
from .denoise import Denoise
from cnn import Denoiser, DnCNN
import torch

class CNNDenoiser(Denoise):
    def __init__(self, cnn_decay=1,
                       cnn_sigma=40, device='cuda'):
        super().__init__()

        # Set user defined parameters
        self.cnn_decay = cnn_decay
        self.cnn_sigma = cnn_sigma
        self.device = device

        self.cnn = Denoiser(net=DnCNN(17), experiment_name='exp_' + str(cnn_sigma), 
                            data=False, sigma=cnn_sigma, batch_size=20).net.to(device)

    def denoise(self, noisy):
        return (noisy - (self.cnn_decay**self.t)*self.cnn(torch.Tensor(noisy)[None][None].to(self.device)).squeeze().detach().cpu().numpy())

### See cnn.py for more info

if __name__=='__main__':
    # Load an image, add noise, try denoising it
    newDenoiser = CNNDenoiser(cnn_decay=1)
