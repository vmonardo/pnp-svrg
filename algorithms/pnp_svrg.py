#!/usr/bin/env python
# coding=utf-8
import time
import numpy as np
from skimage.restoration import estimate_sigma

tol = 1e-5
def pnp_svrg(problem, denoiser, eta, tt, T2, mini_batch_size, verbose=True, lr_decay=1, converge_check=True, diverge_check=False):
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []
    gradient_time = 0
    denoise_time = 0
    
    # Main PnP-SVRG routine
    z = np.copy(problem.Xinit)

    i = 0

    elapsed = time.time()
    time_per_iter.append(time.time() - elapsed)
    psnr_per_iter.append(problem.PSNR(z))

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
        psnr_per_iter.append(problem.PSNR(z))

        # inner loop
        for j in range(T2): 
            if (time.time() - elapsed) >= tt:
                break

            # start PSNR track
            start_PSNR = problem.PSNR(z)

            # start gradient timing
            grad_start_time = time.time()

            # calculate stochastic variance-reduced gradient (SVRG)
            mini_batch = problem.select_mb(mini_batch_size)
            # v = (problem.grad_stoch(z, mini_batch) - problem.grad_stoch(w, mini_batch)) / mini_batch_size + mu
            v = mu

            # Gradient update
            z -= (eta*lr_decay**i)*v

            # end gradient timing
            grad_end_time = time.time() - grad_start_time
            gradient_time += grad_end_time

            if verbose:
                print(str(i) + " " + str(j) + " Before denoising:  " + str(problem.PSNR(z)))

            # start denoising timing
            denoise_start_time = time.time()

            # denoise
            z0 = np.copy(z).reshape(problem.H, problem.W)
            sigma_est = estimate_sigma(z0, multichannel=True, average_sigmas=True)
            z0 = denoiser.denoise(noisy=z0, sigma_est=sigma_est)
            
            # end denoising timing
            denoise_end_time = time.time() - denoise_start_time
            denoise_time += denoise_end_time

            # Log timing
            time_per_iter.append(grad_end_time + denoise_end_time)
            psnr_per_iter.append(problem.PSNR(z0))

            z = np.copy(z0).ravel()

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
        'denoise_time': denoise_time,
        'algo_name': 'PnP SVRG'
    }

def tune_pnp_svrg(args, problem, denoiser, tt, lr_decay=1, verbose=False, converge_check=True, diverge_check=True):
    from hyperopt import STATUS_OK
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
        'loss': (problem.PSNR(problem.Xinit) - problem.PSNR(result['z'])),
        'status': STATUS_OK,
        'algo_name': result['algo_name'],
        'z': result['z'],
        'time_per_iter': result['time_per_iter'],
        'psnr_per_iter': result['psnr_per_iter'],
        'gradient_time': result['gradient_time'],
        'denoise_time': result['denoise_time']
    }
