#!/usr/bin/env python
# coding=utf-8
from denoiser import Denoise
from bm3d import bm3d

class BM3DDenoiser(Denoise):
    def __init__(self, filter_decay=1,
                       noise_est=0):
        super().__init__()

        # Set user defined parameters
        self.filter_decay = filter_decay
        self.noise_est = noise_est

    def denoise(self, noisy, sigma_est=0):
        if sigma_est > 0:
            return bm3d(noisy, sigma_est)
        else:
            return bm3d(noisy, self.noise_est*self.filter_decay**self.t)

### For documentation, see:
# https://webpages.tuni.fi/foi/GCF-BM3D/
# https://pypi.org/project/bm3d/
# https://en.wikipedia.org/wiki/Block-matching_and_3D_filtering

if __name__=='__main__':
    # Load an image, add noise, try denoising it
    newDenoiser = BM3DDenoiser(filter_decay=1, noise_est=1.0)