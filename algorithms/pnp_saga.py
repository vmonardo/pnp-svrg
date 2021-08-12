#!/usr/bin/env python
# coding=utf-8
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

def pnp_saga(problem, denoiser, eta, tt, mini_batch_size, hist_size=50, verbose=True):
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []

    # Main PnP SAGA routine
    z = np.copy(problem.noisy)
    zs = [z]

    denoiser.t = 0

    i = 0

    w = np.copy(z)

    elapsed = time.time()
    
    # calculate stoch_grad
    start_time = time.time()
    
    stoch_init = problem.stoch_grad(z, mini_batch_size)
    grad_history = [stoch_init,]*hist_size
    prev_stoch = stoch_init
    
    time_per_iter.append(time.time() - start_time)
    
    psnr_per_iter.append(peak_signal_noise_ratio(problem.original, z))
    
    while (time.time() - elapsed) < tt:
        # start timing
        start_time = time.time()

        # calculate stochastic gradient
        rand_ind = np.random.choice(hist_size, 1).item()
        grad_history[rand_ind] = problem.stoch_grad(z, mini_batch_size)
        
        v = grad_history[rand_ind] - prev_stoch + sum(grad_history)/hist_size

        # Gradient update
        z -= (eta*problem.lr_decay**denoiser.t)*v

        # Denoise
        z = denoiser.denoise(noisy=z)
        
        # Update prev_stoch
        prev_stoch = grad_history[rand_ind]

        zs.append(z)

        denoiser.t += 1

        # Log timing
        time_per_iter.append(time.time() - start_time)

        psnr_per_iter.append(peak_signal_noise_ratio(problem.original, z))

        if verbose:
            print(str(i) + " " + str(psnr_per_iter[-1]))

        i += 1

    return z, time_per_iter, psnr_per_iter, zs