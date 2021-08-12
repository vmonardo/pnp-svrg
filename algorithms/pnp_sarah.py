#!/usr/bin/env python
# coding=utf-8
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

def pnp_sarah(problem, denoiser, eta, tt, T2, mini_batch_size, verbose=True):
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

        # Initialize ``step 0'' points
        w_previous = np.copy(z) 
        v_previous = problem.full_grad(z)
        
        # General ``step 1'' point
        w_next = w_previous - eta*v_previous

        time_per_iter.append(time.time() - start_time)
        psnr_per_iter.append(peak_signal_noise_ratio(problem.original, z))

        # inner loop
        for j in range(T2): 
            if (time.time() - elapsed) >= tt:
                return z, time_per_iter, psnr_per_iter, zs

            # start timing
            start_time = time.time()

            # calculate recursive stochastic variance-reduced gradient
            v_next = problem.stoch_grad(w_next, mini_batch_size) - problem.stoch_grad(w_previous, mini_batch_size) + v_previous

            # Gradient update
            z -= (eta*problem.lr_decay**denoiser.t)*v_next

            if verbose:
                print("After gradient update: " + str(i) + " " + str(j) + " " + str(peak_signal_noise_ratio(problem.original, z)))

            # Denoise
            z = denoiser.denoise(noisy=z)
            
            zs.append(z)

            denoiser.t += 1

            # update recursion points
            v_previous = np.copy(v_next)
            w_previous = np.copy(z) 
            
            # stop timing
            time_per_iter.append(time.time() - start_time)
            psnr_per_iter.append(peak_signal_noise_ratio(problem.original, z))

            if verbose:
                print("After denoising update: " + str(i) + " " + str(j) + " " + str(psnr_per_iter[-1]))
                print()
        
        i += 1

    # output denoised image, time stats, psnr stats
    return z, time_per_iter, psnr_per_iter, zs