#!/usr/bin/env python
# coding=utf-8
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

def pnp_gd(problem, denoiser, eta, tt, verbose=True):
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []

    # Main PnP GD routine
    z = np.copy(problem.noisy)
    zs = [z]

    denoiser.t = 0

    i = 0

    elapsed = time.time()

    while (time.time() - elapsed) < tt:
        # start timing
        start_time = time.time()

        # calculate full gradient
        v = problem.full_grad(z)

        # Gradient update
        z -= (eta*problem.lr_decay**denoiser.t)*v

        if verbose:
            print(str(i) + " Before denoising:  " + str(peak_signal_noise_ratio(problem.original, z)))
            
        # Denoise
        z = denoiser.denoise(noisy=z)
        
        zs.append(z)

        denoiser.t += 1

        # Log timing
        time_per_iter.append(time.time() - start_time)

        psnr_per_iter.append(peak_signal_noise_ratio(problem.original, z))

        if verbose:
            print(str(i) + " After denoising:  " + str(psnr_per_iter[-1]))

        i += 1

    return z, time_per_iter, psnr_per_iter, zs