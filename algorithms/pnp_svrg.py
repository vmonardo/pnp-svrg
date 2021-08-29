#!/usr/bin/env python
# coding=utf-8
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

tol = 1e-5
def pnp_svrg(problem, denoiser, eta, tt, T2, mini_batch_size, verbose=True, converge_check=True, diverge_check=False):
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []
    gradient_time = 0
    denoise_time = 0
    
    # Main PnP-SVRG routine
    z = np.copy(problem.Xinit)
    zs = [z]

    denoiser.t = 0

    i = 0

    elapsed = time.time()

    # outer loop
    while (time.time() - elapsed) < tt:
        start_time = time.time()

        # Full gradient at reference point
        mu = problem.grad_full(z)

        # Initialize reference point
        w = np.copy(z) 

        time_per_iter.append(time.time() - start_time)
        psnr_per_iter.append(peak_signal_noise_ratio(problem.original, z))

        # inner loop
        for j in range(T2): 
            if (time.time() - elapsed) >= tt:
                return z, time_per_iter, psnr_per_iter, zs

            # start PSNR track
            start_PSNR = peak_signal_noise_ratio(problem.X.reshape(problem.H,problem.W), z)

            # start gradient timing
            grad_start_time = time.time()

            # calculate stochastic variance-reduced gradient (SVRG)
            mini_batch = problem.batch(mini_batch_size)
            v = (problem.grad_stoch(z, mini_batch) - problem.grad_stoch(w, mini_batch)) / mini_batch_size + mu

            # Gradient update
            z -= (eta*problem.lr_decay**denoiser.t)*v

            # end gradient timing
            grad_end_time = time.time() - grad_start_time
            gradient_time += grad_end_time

            if verbose:
                print(str(i) + " Before denoising:  " + str(peak_signal_noise_ratio(problem.X.reshape(problem.H,problem.W), z)))

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
            psnr_per_iter.append(peak_signal_noise_ratio(problem.X.reshape(problem.H,problem.W), z))

            if verbose:
                print("After denoising update: " + str(i) + " " + str(j) + " " + str(psnr_per_iter[-1]))

            # Check convergence in terms of PSNR
            if converge_check is True and np.abs(start_PSNR - psnr_per_iter[-1]) < tol:
                break

            # Check divergence of PSNR
            if diverge_check is True and psnr_per_iter[-1] < 0:
                break
        
        i += 1

    # output denoised image, time stats, psnr stats
    return z, time_per_iter, psnr_per_iter, zs, gradient_time, denoise_time