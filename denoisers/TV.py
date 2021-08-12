#!/usr/bin/env python
# coding=utf-8
from .denoise import Denoise
from skimage.restoration import denoise_wavelet

class TVDenoiser(Denoise):
    def __init__(self, multi=True, rescale_sigma=True):
        super().__init__()

        # Set user defined parameters
        self.multi = multi
        self.rescale_sigma = rescale_sigma

    def denoise(self, noisy):
        return denoise_wavelet(noisy, multichannel=self.multi, rescale_sigma=self.rescale_sigma)

### For documentation, see:
# https://scikit-image.org/docs/dev/api/skimage.restoration.html#skimage.restoration.denoise_wavelet
# https://en.wikipedia.org/wiki/Total_variation_denoising

if __name__=='__main__':
    # Load an image, add noise, try denoising it
    newDenoiser = TVDenoiser()
