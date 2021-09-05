#!/usr/bin/env python
# coding=utf-8
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.restoration import estimate_sigma
from hyperopt import STATUS_OK

tol = 1e-5
def pnp_sgd(problem, denoiser, eta, tt, mini_batch_size, verbose=True, lr_decay=1, converge_check=True, diverge_check=False):
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []
    gradient_time = 0
    denoise_time = 0

    # Main PnP SGD routine
    z = np.copy(problem.Xinit)

    denoiser.t = 0

    i = 0

    elapsed = time.time()

    while (time.time() - elapsed) < tt:
        # start PSNR track
        start_PSNR = peak_signal_noise_ratio(problem.X.reshape(problem.H,problem.W), z.reshape(problem.H,problem.W))

        # start gradient timing
        grad_start_time = time.time()

        # calculate stochastic gradient
        mini_batch = problem.select_mb(mini_batch_size)
        v = problem.grad_stoch(z, mini_batch) / mini_batch_size

        # Gradient update
        z = z.reshape(problem.H,problem.W)
        z -= (eta*lr_decay**denoiser.t)*v.reshape(problem.H,problem.W)

        # end gradient timing
        grad_end_time = time.time() - grad_start_time
        gradient_time += grad_end_time

        if verbose:
            print(str(i) + " Before denoising:  " + str(peak_signal_noise_ratio(problem.X.reshape(problem.H,problem.W), z.reshape(problem.H,problem.W))))

        # start denoising timing
        denoise_start_time = time.time()

        # Denoise
        z = denoiser.denoise(noisy=z, sigma_est=0)

        # end denoising timing
        denoise_end_time = time.time() - denoise_start_time
        denoise_time += denoise_end_time

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

    # output denoised image, time stats, psnr stats
    return {
        'z': z,
        'time_per_iter': time_per_iter,
        'psnr_per_iter': psnr_per_iter,
        'gradient_time': gradient_time,
        'denoise_time': denoise_time
    }
    # return z, time_per_iter, psnr_per_iter, zs, gradient_time, denoise_time

def tune_pnp_sgd(args, problem, denoiser, tt, verbose=True, lr_decay=1, converge_check=True, diverge_check=False):
    eta, mini_batch_size, dstrength = args
    denoiser.sigma_est = dstrength
    result = pnp_sgd(   eta=eta,
                        mini_batch_size=mini_batch_size,
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
    # return z, time_per_iter, psnr_per_iter, zs, gradient_time, denoise_time