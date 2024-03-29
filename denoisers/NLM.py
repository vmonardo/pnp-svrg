#!/usr/bin/env python
# coding=utf-8
try:
    from .denoiser import Denoise
except:
    from denoiser import Denoise
from skimage.restoration import denoise_nl_means

class NLMDenoiser(Denoise):
    def __init__(self, decay=1,
                       denoise_strength=0, patch_size=4, patch_distance=5, 
                       sigma_modifier=1, fast_mode=False, multichannel=True):
        super().__init__()

        # Set user defined parameters
        self.decay = decay
        self.denoise_strength = denoise_strength
        self.fast_mode = fast_mode
        self.sigma_modifier = sigma_modifier
        self.patch = dict(patch_size=patch_size, patch_distance=patch_distance, multichannel=multichannel)

    def denoise(self, noisy, sigma_est=0):
        self.t += 1
        if self.sigma > 0:
            return denoise_nl_means(noisy, h=sigma_est*self.sigma_modifier, sigma=sigma_est*self.sigma_modifier, fast_mode=self.fast_mode, **self.patch)
        else:
            return denoise_nl_means(noisy, h=self.denoise_strength*self.decay**self.t, fast_mode=self.fast_mode, **self.patch)


### For documentation, see:
# https://scikit-image.org/docs/dev/api/skimage.restoration.html#skimage.restoration.denoise_nl_means

if __name__=='__main__':
    # Demo modified from:
    # https://scikit-image.org/docs/dev/auto_examples/filters/plot_nonlocal_means.html
    import numpy as np
    import matplotlib.pyplot as plt

    from skimage import data, img_as_float
    from skimage.restoration import denoise_nl_means, estimate_sigma
    from skimage.metrics import peak_signal_noise_ratio
    from skimage.util import random_noise


    astro = img_as_float(data.astronaut())
    astro = astro[30:180, 150:300]

    sigma = 0.08
    noisy = random_noise(astro, var=sigma**2)

    # estimate the noise standard deviation from the noisy image
    sigma_est = estimate_sigma(noisy, multichannel=True, average_sigmas=True)
    print(f"estimated noise standard deviation = {sigma_est}")

    patch_kw = dict(patch_size=5,      # 5x5 patches
                    patch_distance=6,  # 13x13 search area
                    multichannel=True)

    # slow algorithm
    NLMslow = NLMDenoiser(decay=1, sigma_est=1.15 * sigma_est, patch_size=5, fast_mode=False, patch_distance=6)
    denoise = NLMslow.denoise(noisy)

    # slow algorithm, sigma provided
    NLMslowwithsigma = NLMDenoiser(decay=1, sigma_est=0.8 * sigma_est, sigma=sigma_est, patch_size=5, fast_mode=False, patch_distance=6)
    denoise2 = NLMslowwithsigma.denoise(noisy)

    # fast algorithm
    NLMfast = NLMDenoiser(decay=1, sigma_est=0.8 * sigma_est, patch_size=5, fast_mode=True, patch_distance=6)
    denoise_fast = NLMfast.denoise(noisy)

    # fast algorithm, sigma provided
    NLMfastwithsigma = NLMDenoiser(decay=1, sigma_est=0.6 * sigma_est, sigma=sigma_est, patch_size=5, fast_mode=True, patch_distance=6)
    denoise2_fast = NLMfastwithsigma.denoise(noisy)

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 6),
                        sharex=True, sharey=True)

    ax[0, 0].imshow(noisy)
    ax[0, 0].axis('off')
    ax[0, 0].set_title('noisy')
    ax[0, 1].imshow(denoise)
    ax[0, 1].axis('off')
    ax[0, 1].set_title('non-local means\n(slow)')
    ax[0, 2].imshow(denoise2)
    ax[0, 2].axis('off')
    ax[0, 2].set_title('non-local means\n(slow, using $\\sigma_{est}$)')
    ax[1, 0].imshow(astro)
    ax[1, 0].axis('off')
    ax[1, 0].set_title('original\n(noise free)')
    ax[1, 1].imshow(denoise_fast)
    ax[1, 1].axis('off')
    ax[1, 1].set_title('non-local means\n(fast)')
    ax[1, 2].imshow(denoise2_fast)
    ax[1, 2].axis('off')
    ax[1, 2].set_title('non-local means\n(fast, using $\\sigma_{est}$)')

    fig.tight_layout()

    # print PSNR metric for each case
    psnr_noisy = peak_signal_noise_ratio(astro, noisy)
    psnr = peak_signal_noise_ratio(astro, denoise)
    psnr2 = peak_signal_noise_ratio(astro, denoise2)
    psnr_fast = peak_signal_noise_ratio(astro, denoise_fast)
    psnr2_fast = peak_signal_noise_ratio(astro, denoise2_fast)

    print(f"PSNR (noisy) = {psnr_noisy:0.2f}")
    print(f"PSNR (slow) = {psnr:0.2f}")
    print(f"PSNR (slow, using sigma) = {psnr2:0.2f}")
    print(f"PSNR (fast) = {psnr_fast:0.2f}")
    print(f"PSNR (fast, using sigma) = {psnr2_fast:0.2f}")

    plt.show()