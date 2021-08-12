#!/usr/bin/env python
# coding=utf-8
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

def pnp_lsvrg(problem, denoiser, eta, tt, mini_batch_size, prob_update=0.1, verbose=True):
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []

    # Main PnP LSVRG routine
    z = np.copy(problem.Xinit)
    zs = [z]

    denoiser.t = 0

    i = 0

    w = np.copy(z)

    elapsed = time.time()
    
    # calculate full gradient
    start_time = time.time()
    mu = problem.full_grad(z)
    time_per_iter.append(time.time() - start_time)
    psnr_per_iter.append(peak_signal_noise_ratio(problem.original, z))

    while (time.time() - elapsed) < tt:
        # start timing
        start_time = time.time()

        # calculate stochastic gradient
        v = problem.stoch_grad(z, mini_batch_size) - problem.stoch_grad(w, mini_batch_size) + mu

        # Gradient update
        z -= (eta*problem.lr_decay**denoiser.t)*v

        # Denoise
        z = denoiser.denoise(noisy=z)

        zs.append(z)

        denoiser.t += 1

        # update reference point with probability prob_update
        if np.random.random() < prob_update:
            w = np.copy(z)
            mu = problem.full_grad(z)

        # Log timing
        time_per_iter.append(time.time() - start_time)

        psnr_per_iter.append(peak_signal_noise_ratio(problem.original, z))

        if verbose:
            print(str(i) + " " + str(psnr_per_iter[-1]))

        i += 1

    return z, time_per_iter, psnr_per_iter, zs