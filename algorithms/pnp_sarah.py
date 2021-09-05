#!/usr/bin/env python
# coding=utf-8
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.restoration import estimate_sigma
from hyperopt import STATUS_OK

tol = 1e-5
def pnp_sarah(problem, denoiser, eta, tt, T2, mini_batch_size, verbose=True, lr_decay=1, converge_check=True, diverge_check=False):
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []
    gradient_time = 0
    denoise_time = 0
    
    # Main PnP-SVRG routine
    z = np.copy(problem.Xinit).ravel()
    
    denoiser.t = 0

    i = 0

    elapsed = time.time()

    # outer loop
    break_out_flag = False
    while (time.time() - elapsed) < tt:
        if break_out_flag:
            break
        # Initialize ``step 0'' points
        w_previous = np.copy(z).ravel()

        # start gradient timing
        grad_start_time = time.time()

        v_previous = problem.grad_full(z).ravel()
        
        # General ``step 1'' point
        w_next = w_previous - eta*v_previous

        # end gradient timing
        grad_end_time = time.time() - grad_start_time
        gradient_time += grad_end_time

        # start denoising timing
        denoise_start_time = time.time()

        # make denoising variable
        w_next = w_next.reshape(problem.H, problem.W)
        w_next = denoiser.denoise(noisy=w_next)

        # end denoising timing
        denoise_end_time = time.time() - denoise_start_time
        denoise_time += denoise_end_time

        time_per_iter.append(grad_end_time + denoise_end_time)
        psnr_per_iter.append(peak_signal_noise_ratio(problem.X.reshape(problem.H,problem.W), w_next))

        w_next = w_next.ravel()

        # inner loop
        for j in range(T2): 
            if (time.time() - elapsed) >= tt:
                break
            
            # start PSNR track
            start_PSNR = peak_signal_noise_ratio(problem.X.reshape(problem.H,problem.W), z.reshape(problem.H,problem.W))

            # start gradient timing
            grad_start_time = time.time()

            # calculate recursive stochastic variance-reduced gradient
            mini_batch = problem.select_mb(mini_batch_size)
            v_next = (problem.grad_stoch(w_next, mini_batch).ravel() - problem.grad_stoch(w_previous, mini_batch).ravel()) / mini_batch_size + v_previous.ravel()

            # Gradient update
            z -= (eta*lr_decay**denoiser.t)*v_next

            # end gradient timing
            grad_end_time = time.time() - grad_start_time
            gradient_time += grad_end_time

            if verbose:
                print("After gradient update: " + str(i) + " " + str(j) + " " + str(peak_signal_noise_ratio(problem.X.reshape(problem.H,problem.W), z.reshape(problem.H,problem.W))))

            # start denoising timing
            denoise_start_time = time.time()    

            # Denoise
            z = z.reshape(problem.H, problem.W)
            z = denoiser.denoise(noisy=z)

            # end denoising timing
            denoise_end_time = time.time() - denoise_start_time
            denoise_time += denoise_end_time

            denoiser.t += 1

            # update recursion points
            v_previous = np.copy(v_next)
            w_previous = np.copy(z.ravel()) 
            
            # stop timing
            time_per_iter.append(grad_end_time + denoise_end_time)
            psnr_per_iter.append(peak_signal_noise_ratio(problem.X.reshape(problem.H,problem.W), z))

            z = z.ravel()

            if verbose:
                print("After denoising update: " + str(i) + " " + str(j) + " " + str(psnr_per_iter[-1]))
                print()

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

def tune_pnp_sarah(args, problem, denoiser, tt, verbose=True, lr_decay=1, converge_check=True, diverge_check=False):
    eta, mini_batch_size, T2, dstrength = args
    denoiser.sigma_est = dstrength
    result = pnp_sarah( eta=eta,
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
    # return z, time_per_iter, psnr_per_iter, zs, gradient_time, denoise_time