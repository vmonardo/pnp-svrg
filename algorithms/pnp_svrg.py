#!/usr/bin/env python
# coding=utf-8
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.restoration import estimate_sigma
from hyperopt import STATUS_OK

tol = 1e-5

def pnp_svrg(problem, denoiser, eta, tt, T2, mini_batch_size, verbose=True, lr_decay=1, converge_check=True, diverge_check=False):
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []
    gradient_time = 0
    denoise_time = 0
    
    # Main PnP-SVRG routine
    z = np.copy(problem.Xinit)

    denoiser.t = 0

    i = 0

    elapsed = time.time()

    # outer loop
    break_out_flag = False
    while (time.time() - elapsed) < tt:
        if break_out_flag:
            break
        start_time = time.time()

        # Full gradient at reference point
        mu = problem.grad_full(z)

        # Initialize reference point
        w = np.copy(z) 

        time_per_iter.append(time.time() - start_time)
        psnr_per_iter.append(peak_signal_noise_ratio(problem.X.reshape(problem.H,problem.W), z.reshape(problem.H,problem.W)))

        # inner loop
        for j in range(T2): 
            if (time.time() - elapsed) >= tt:
                break

            # start PSNR track
            start_PSNR = peak_signal_noise_ratio(problem.X.reshape(problem.H,problem.W), z.reshape(problem.H,problem.W))

            # start gradient timing
            grad_start_time = time.time()

            # calculate stochastic variance-reduced gradient (SVRG)
            mini_batch = problem.select_mb(mini_batch_size)
            v = (problem.grad_stoch(z, mini_batch) - problem.grad_stoch(w, mini_batch)) / mini_batch_size + mu

            # Gradient update
            z = z.reshape(problem.H,problem.W)
            z -= (eta*lr_decay**denoiser.t)*v.reshape(problem.H,problem.W)

            # end gradient timing
            grad_end_time = time.time() - grad_start_time
            gradient_time += grad_end_time

            if verbose:
                print(str(i) + str(j) + " Before denoising:  " + str(peak_signal_noise_ratio(problem.X.reshape(problem.H,problem.W), z.reshape(problem.H,problem.W))))

            # start denoising timing
            denoise_start_time = time.time()

            # estimate sigma 
            # sigma_est = estimate_sigma(z, multichannel=True, average_sigmas=True)

            # Denoise
            z = denoiser.denoise(noisy=z)
            
            # end denoising timing
            denoise_end_time = time.time() - denoise_start_time
            denoise_time += denoise_end_time

            denoiser.t += 1

            # Log timing
            time_per_iter.append(grad_end_time + denoise_end_time)
            psnr_per_iter.append(peak_signal_noise_ratio(problem.X.reshape(problem.H,problem.W), z.reshape(problem.H,problem.W)))

            if verbose:
                print("After denoising update: " + str(i) + " " + str(j) + " " + str(psnr_per_iter[-1]))

            # Check convergence in terms of PSNR
            if converge_check is True and np.abs(start_PSNR - psnr_per_iter[-1]) < tol:
                break_out_flag = True
                break
            # Check divergence of PSNR
            if diverge_check is True and psnr_per_iter[-1] < 0:
                break_out_flag = True
                break
        i += 1

    # output denoised image, time stats, psnr stats
    return {
        'z': z,
        'time_per_iter': time_per_iter,
        'psnr_per_iter': psnr_per_iter,
        'gradient_time': gradient_time,
        'denoise_time': denoise_time
    }
    # return z, time_per_iter, psnr_per_iter, zs, gradient_time, denoise_time

def tune_pnp_svrg(args, problem, denoiser, tt, verbose=True, lr_decay=1, converge_check=True, diverge_check=False):
    eta, mini_batch_size, T2, dstrength = args
    denoiser.sigma_est = dstrength
    result = pnp_svrg(  eta=eta,
                        mini_batch_size=mini_batch_size,
                        T2=T2,
                        problem=problem,
                        denoiser=denoiser,
                        tt=tt,
                        verbose=verbose,
                        lr_decay=lr_decay,
                        converge_check=converge_check,
                        diverge_check=diverge_check )

    # output denoised image, time stats, psnr stats
    return {
        'loss': -result['psnr_per_iter'][-1],    # Look for hyperparameters that increase the positive change in PSNR 
        'status': STATUS_OK,
        'z': result['z'],
        'time_per_iter': result['time_per_iter'],
        'psnr_per_iter': result['psnr_per_iter'],
        'gradient_time': result['gradient_time'],
        'denoise_time': result['denoise_time']
    }