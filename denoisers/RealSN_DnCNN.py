#!/usr/bin/env python
# coding=utf-8
import numpy as np
import torch
from denoisers.DeepDenoisers.utils.utils import load_model
from denoisers.denoiser import Denoise

class RealSN_DnCNNDenoiser(Denoise):
    def __init__(self, model_type, sigma):
        super().__init__()

        self.model_type = model_type
        self.sigma = sigma
        self.model = load_model(self.model_type, self.sigma)

    def denoise(self, noisy, sigma_est=0):
        """ Denoising step. """
        m, n = noisy.shape
        xtilde = np.copy(noisy)
        mintmp = np.min(xtilde)
        maxtmp = np.max(xtilde)
        xtilde = (xtilde - mintmp) / (maxtmp - mintmp)
        
        # the reason for the following scaling:
        # our denoisers are trained with "normalized images + noise"
        # so the scale should be 1 + O(sigma)
        scale_range = 1.0 + sigma_est/255.0/2.0
        scale_shift = (1 - scale_range) / 2.0
        xtilde = xtilde * scale_range + scale_shift

        # pytorch denoising model
        xtilde_torch = np.reshape(xtilde, (1,1,m,n))
        xtilde_torch = torch.from_numpy(xtilde_torch).type(torch.FloatTensor).cuda()
        r = self.model(xtilde_torch).cpu().numpy()
        r = np.reshape(r, (m,n))
        x = xtilde - r

        # scale and shift the denoised image back
        x = (x - scale_shift) / scale_range
        x = x * (maxtmp - mintmp) + mintmp

        return x
