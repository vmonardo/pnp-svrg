#!/usr/bin/env python
# coding=utf-8
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

tol = 1e-5
def pnp_gd(problem, denoiser, eta, tt, verbose=True, converge_check=True, diverge_check=False):
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []
    gradient_time = 0
    denoise_time = 0

    # Main PnP GD routine
    z = np.copy(problem.Xinit)
    zs = [z]

    denoiser.t = 0

    i = 0

    elapsed = time.time()

    while (time.time() - elapsed) < tt:
        # start PSNR track
        start_PSNR = peak_signal_noise_ratio(problem.X.reshape(problem.H,problem.W), z.reshape(problem.H,problem.W))

        # start gradient timing
        grad_start_time = time.time()

        # calculate full gradient
        v = problem.grad_full(z)

        # Gradient update
        z -= (eta*problem.lr_decay**denoiser.t)*v

        # end gradient timing
        grad_end_time = time.time() - grad_start_time
        gradient_time += grad_end_time

        if verbose:
            print(str(i) + " Before denoising:  " + str(peak_signal_noise_ratio(problem.X.reshape(problem.H,problem.W), z.reshape(problem.H,problem.W))))
            
        # start denoising timing
        denoise_start_time = time.time()

        # Denoise
        z = denoiser.denoise(noisy=z)
        
        # end denoising timing
        denoise_end_time = time.time() - denoise_start_time
        denoise_time += denoise_end_time

        zs.append(z)

        denoiser.t += 1

        # Log timing
        time_per_iter.append(grad_end_time + denoise_end_time)

        psnr_per_iter.append(peak_signal_noise_ratio(problem.X.reshape(problem.H,problem.W), z.reshape(problem.H,problem.W)))

        if verbose:
            print(str(i) + " After denoising:  " + str(psnr_per_iter[-1]))

        i += 1

        # Check convergence in terms of PSNR
        if converge_check is True and np.abs(start_PSNR - psnr_per_iter[-1]) < tol:
            break

        # Check divergence of PSNR
        if diverge_check is True and psnr_per_iter[-1] < 0:
            break


    return z, time_per_iter, psnr_per_iter, zs, gradient_time, denoise_time