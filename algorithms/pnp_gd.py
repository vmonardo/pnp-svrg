#!/usr/bin/env python
# coding=utf-8
import time
import numpy as np

tol = 1e-5
def pnp_gd(problem, denoiser, eta, tt, verbose=True, lr_decay=1, converge_check=True, diverge_check=False):
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []
    gradient_time = 0
    denoise_time = 0

    # Main PnP GD routine
    z = np.copy(problem.Xinit)

    i = 0

    elapsed = time.time()

    while (time.time() - elapsed) < tt:
        # start PSNR track
        start_PSNR = problem.PSNR(z)

        # start gradient timing
        grad_start_time = time.time()

        # calculate full gradient
        v = problem.grad_full(z)

        # Gradient update
        z -= (eta*lr_decay**i)*v

        # end gradient timing
        grad_end_time = time.time() - grad_start_time
        gradient_time += grad_end_time

        if verbose:
            print(str(i) + " Before denoising:  " + str(problem.PSNR(z)))

        # start denoising timing
        denoise_start_time = time.time()

        # denoise
        z0 = np.copy(z).reshape(problem.H, problem.W)
        z0 = denoiser.denoise(noisy=z0)
        
        # end denoising timing
        denoise_end_time = time.time() - denoise_start_time
        denoise_time += denoise_end_time

        # Logging
        time_per_iter.append(grad_end_time + denoise_end_time)
        psnr_per_iter.append(problem.PSNR(z0))

        z = np.copy(z0).ravel()

        if verbose:
            print(str(i) + " After denoising:  " + str(psnr_per_iter[-1]))

        i += 1

        # Check convergence in terms of PSNR
        if converge_check is True and np.abs(start_PSNR - psnr_per_iter[-1]) < tol:
            break

        # Check divergence of PSNR
        if diverge_check is True and psnr_per_iter[-1] < 0:
            break

    # output denoised image, time stats, psnr stats
    return {
        'z': z,
        'time_per_iter': time_per_iter,
        'psnr_per_iter': psnr_per_iter,
        'gradient_time': gradient_time,
        'denoise_time': denoise_time,
        'algo_name': 'pnp_gd'
    }

def tune_pnp_gd(args, problem, denoiser, tt, lr_decay=1, verbose=False, converge_check=True, diverge_check=True):
    from hyperopt import STATUS_OK
    eta, dstrength = args
    denoiser.sigma_est = dstrength
    result = pnp_gd(    eta=eta,
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
        'denoise_time': result['denoise_time'],
        'algo_name': result['algo_name']
    }