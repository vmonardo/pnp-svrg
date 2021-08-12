#!/usr/bin/env python
# coding=utf-8
from .denoise import Denoise
from skimage.restoration import denoise_nl_means

class NLMDenoiser(Denoise):
    def __init__(self, filter_decay=1,
                       filter_size=0, patch_size=0, patch_distance=0, multichannel=True):
        super().__init__()

        # Set user defined parameters
        self.filter_decay = filter_decay
        self.filter_size = filter_size
        self.patch = dict(patch_size=patch_size, patch_distance=patch_distance, multichannel=multichannel)

    def denoise(self, noisy):
        return denoise_nl_means(noisy, h=self.filter_size*self.filter_decay**self.t, fast_mode=True, **self.patch)


### For documentation, see:
# https://scikit-image.org/docs/dev/auto_examples/filters/plot_nonlocal_means.html

if __name__=='__main__':
    # Load an image, add noise, try denoising it
    newDenoiser = NLMDenoiser(filter_decay=1, filter_size=1.0, patch_size=5, patch_distance=6)
