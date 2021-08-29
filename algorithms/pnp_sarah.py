#!/usr/bin/env python
# coding=utf-8
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

tol = 1e-5
def pnp_sarah(problem, denoiser, eta, tt, T2, mini_batch_size, verbose=True, converge_check=True, diverge_check=False):
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
        # Initialize ``step 0'' points
        w_previous = np.copy(z) 

        # start gradient timing
        grad_start_time = time.time()

        v_previous = problem.grad_full(z)
        
        # General ``step 1'' point
        w_next = w_previous - eta*v_previous

        # end gradient timing
        grad_end_time = time.time() - grad_start_time
        gradient_time += grad_end_time

        # start denoising timing
        denoise_start_time = time.time()

        w_next = denoiser.denoise(noisy=w_next)

        # end denoising timing
        denoise_end_time = time.time() - denoise_start_time
        denoise_time += denoise_end_time

        time_per_iter.append(grad_end_time + denoise_end_time)
        psnr_per_iter.append(peak_signal_noise_ratio(problem.X.reshape(problem.H,problem.W), w_next))

        # inner loop
        for j in range(T2): 
            if (time.time() - elapsed) >= tt:
                return z, time_per_iter, psnr_per_iter, zs
            
            # start PSNR track
            start_PSNR = peak_signal_noise_ratio(problem.X.reshape(problem.H,problem.W), z.reshape(problem.H,problem.W))

            # start gradient timing
            grad_start_time = time.time()

            # calculate recursive stochastic variance-reduced gradient
            mini_batch = problem.select_mb(mini_batch_size)
            v_next = (problem.grad_stoch(w_next, mini_batch) - problem.grad_stoch(w_previous, mini_batch)) / mini_batch_size + v_previous

            # Gradient update
            z -= (eta*problem.lr_decay**denoiser.t)*v_next

            # end gradient timing
            grad_end_time = time.time() - grad_start_time
            gradient_time += grad_end_time

            if verbose:
                print("After gradient update: " + str(i) + " " + str(j) + " " + str(peak_signal_noise_ratio(problem.X.reshape(problem.H,problem.W), z.reshape(problem.H,problem.W))))

            # start denoising timing
            denoise_start_time = time.time()    

            # Denoise
            z = denoiser.denoise(noisy=z)

            # end denoising timing
            denoise_end_time = time.time() - denoise_start_time
            denoise_time += denoise_end_time
            
            zs.append(z)

            denoiser.t += 1

            # update recursion points
            v_previous = np.copy(v_next)
            w_previous = np.copy(z) 
            
            # stop timing
            time_per_iter.append(grad_end_time + denoise_end_time)
            psnr_per_iter.append(peak_signal_noise_ratio(problem.X.reshape(problem.H,problem.W), z.reshape(problem.H,problem.W)))

            if verbose:
                print("After denoising update: " + str(i) + " " + str(j) + " " + str(psnr_per_iter[-1]))
                print()

            # Check convergence in terms of PSNR
            if converge_check is True and np.abs(start_PSNR - psnr_per_iter[-1]) < tol:
                break

            # Check divergence of PSNR
            if diverge_check is True and psnr_per_iter[-1] < 0:
                break
        
        i += 1

    # output denoised image, time stats, psnr stats
    return z, time_per_iter, psnr_per_iter, zs, gradient_time, denoise_time