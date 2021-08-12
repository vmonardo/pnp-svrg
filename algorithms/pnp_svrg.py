#!/usr/bin/env python
# coding=utf-8
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

def pnp_svrg(problem, denoiser, eta, tt, T2, mini_batch_size, verbose=True):
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []
    
    # Main PnP-SVRG routine
    z = np.copy(problem.noisy)
    zs = [z]

    denoiser.t = 0

    i = 0

    elapsed = time.time()

    # outer loop
    while (time.time() - elapsed) < tt:
        start_time = time.time()

        # Full gradient at reference point
        mu = problem.full_grad(z)

        # Initialize reference point
        w = np.copy(z) 

        time_per_iter.append(time.time() - start_time)
        psnr_per_iter.append(peak_signal_noise_ratio(problem.original, z))

        # inner loop
        for j in range(T2): 
            if (time.time() - elapsed) >= tt:
                return z, time_per_iter, psnr_per_iter, zs

            # start timing
            start_time = time.time()

            # calculate stochastic variance-reduced gradient (SVRG)
            v = (problem.stoch_grad(z, mini_batch_size) - problem.stoch_grad(w, mini_batch_size)) + mu

            # Gradient update
            z -= (eta*problem.lr_decay**denoiser.t)*v

            if verbose:
                print("After gradient update: " + str(i) + " " + str(j) + " " + str(peak_signal_noise_ratio(problem.original, z)))

            # Denoise
            z = denoiser.denoise(noisy=z)
            
            zs.append(z)

            denoiser.t += 1

            # stop timing
            time_per_iter.append(time.time() - start_time)
            psnr_per_iter.append(peak_signal_noise_ratio(problem.original, z))

            if verbose:
                print("After denoising update: " + str(i) + " " + str(j) + " " + str(psnr_per_iter[-1]))
                print()
        
        i += 1

    # output denoised image, time stats, psnr stats
    return z, time_per_iter, psnr_per_iter, zs