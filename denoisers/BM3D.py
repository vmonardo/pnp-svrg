#!/usr/bin/env python
# coding=utf-8
try:
    from .denoiser import Denoise
except:
    from denoiser import Denoise
from bm3d import bm3d

class BM3DDenoiser(Denoise):
    def __init__(self, decay=1,
                       denoise_strength=0,
                       sigma_modifier=1):
        super().__init__()

        # Set user defined parameters
        self.decay = decay
        self.denoise_strength = denoise_strength
        self.sigma_modifier = sigma_modifier

    def denoise(self, noisy, sigma_est=0):
        self.t += 1
        if sigma_est > 0:
            return bm3d(noisy, self.sigma_modifier*sigma_est)
        else:
            return bm3d(noisy, self.denoise_strength*self.decay**self.t)
        

### For documentation, see:
# https://webpages.tuni.fi/foi/GCF-BM3D/
# https://pypi.org/project/bm3d/
# https://en.wikipedia.org/wiki/Block-matching_and_3D_filtering

if __name__=='__main__':
    # Load an image, add noise, try denoising it
    newDenoiser = BM3DDenoiser(decay=1, sigma_est=1.0)