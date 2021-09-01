#!/usr/bin/env python
# coding=utf-8
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.restoration import estimate_sigma
from hyperopt import STATUS_OK

tol = 1e-5
def pnp_saga(problem, denoiser, eta, tt, mini_batch_size, hist_size=50, verbose=True, lr_decay=1, converge_check=True, diverge_check=False):
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []
    gradient_time = 0
    denoise_time = 0

    # Main PnP SAGA routine
    z = np.copy(problem.Xinit)
    zs = [z]

    denoiser.t = 0

    i = 0

    w = np.copy(z)

    elapsed = time.time()
    
    # calculate stoch_grad
    start_time = time.time()
    
    mini_batch = problem.select_mb(mini_batch_size)
    stoch_init = problem.grad_stoch(z, mini_batch)
    
    grad_history = [stoch_init,]*hist_size
    prev_stoch = stoch_init
    
    time_per_iter.append(time.time() - start_time)
    
    psnr_per_iter.append(peak_signal_noise_ratio(problem.X, z))
    
    while (time.time() - elapsed) < tt:
        # start PSNR track
        start_PSNR = peak_signal_noise_ratio(problem.X.reshape(problem.H,problem.W), z.reshape(problem.H,problem.W))

        # start gradient timing
        grad_start_time = time.time()

        # calculate stochastic gradient
        mini_batch = problem.select_mb(mini_batch_size)
        rand_ind = np.random.choice(hist_size, 1).item()
        grad_history[rand_ind] = problem.grad_stoch(z, mini_batch) / mini_batch_size
        
        v = grad_history[rand_ind] - prev_stoch + sum(grad_history)/hist_size

        # Gradient update
        z = z.reshape(problem.H,problem.W)
        z -= (eta*lr_decay**denoiser.t)*v.reshape(problem.H,problem.W)

        # end gradient timing
        grad_end_time = time.time() - grad_start_time
        gradient_time += grad_end_time

        # start denoising timing
        denoise_start_time = time.time()

        if verbose:
            print(str(i) + " Before denoising:  " + str(peak_signal_noise_ratio(problem.X.reshape(problem.H,problem.W), z.reshape(problem.H,problem.W))))

        # estimate sigma 
        sigma_est = estimate_sigma(z, multichannel=True, average_sigmas=True)

        # Denoise
        z = denoiser.denoise(noisy=z, true_sigma=sigma_est)

        # end denoising timing
        denoise_end_time = time.time() - denoise_start_time
        denoise_time += denoise_end_time
        
        # Update prev_stoch
        prev_stoch = grad_history[rand_ind]

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

    # output denoised image, time stats, psnr stats
    return {
        'loss': -peak_signal_noise_ratio(problem.X.reshape(problem.H,problem.W), z.reshape(problem.H,problem.W)),
        'status': STATUS_OK,
        'z': z,
        'time_per_iter': time_per_iter,
        'psnr_per_iter': psnr_per_iter,
        'zs': zs,
        'gradient_time': gradient_time,
        'denoise_time': denoise_time
    }
    # return z, time_per_iter, psnr_per_iter, zs, gradient_time, denoise_time

def tune_pnp_saga(args, problem, denoiser, tt, hist_size=50, verbose=True, lr_decay=1, converge_check=True, diverge_check=False):
    eta, mini_batch_size, dstrength = args 
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []
    gradient_time = 0
    denoise_time = 0

    # Main PnP SAGA routine
    z = np.copy(problem.Xinit)
    zs = [z]

    denoiser.t = 0

    i = 0

    w = np.copy(z)

    elapsed = time.time()
    
    # calculate stoch_grad
    start_time = time.time()
    
    mini_batch = problem.select_mb(mini_batch_size)
    stoch_init = problem.grad_stoch(z, mini_batch)
    grad_history = [stoch_init,]*hist_size
    prev_stoch = stoch_init
    
    time_per_iter.append(time.time() - start_time)
    
    psnr_per_iter.append(peak_signal_noise_ratio(problem.X, z))
    
    while (time.time() - elapsed) < tt:
        # start PSNR track
        start_PSNR = peak_signal_noise_ratio(problem.X.reshape(problem.H,problem.W), z.reshape(problem.H,problem.W))

        # start gradient timing
        grad_start_time = time.time()

        # calculate stochastic gradient
        mini_batch = problem.select_mb(mini_batch_size)
        rand_ind = np.random.choice(hist_size, 1).item()
        grad_history[rand_ind] = problem.grad_stoch(z, mini_batch) / mini_batch_size
        
        v = grad_history[rand_ind] - prev_stoch + sum(grad_history)/hist_size

        # Gradient update
        z = z.reshape(problem.H,problem.W)
        z -= (eta*lr_decay**denoiser.t)*v.reshape(problem.H,problem.W)

        # end gradient timing
        grad_end_time = time.time() - grad_start_time
        gradient_time += grad_end_time

        # start denoising timing
        denoise_start_time = time.time()

        if verbose:
            print(str(i) + " Before denoising:  " + str(peak_signal_noise_ratio(problem.X.reshape(problem.H,problem.W), z.reshape(problem.H,problem.W))))

        # estimate sigma 
        # sigma_est = estimate_sigma(z, multichannel=True, average_sigmas=True)

        # Denoise
        z = denoiser.denoise(noisy=z, sigma_est=dstrength)

        # end denoising timing
        denoise_end_time = time.time() - denoise_start_time
        denoise_time += denoise_end_time
        
        # Update prev_stoch
        prev_stoch = grad_history[rand_ind]

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

    # output denoised image, time stats, psnr stats
    return {
        'loss': -psnr_per_iter[-1],    # Look for hyperparameters that increase the positive change in PSNR
        'status': STATUS_OK,
        'z': z,
        'time_per_iter': time_per_iter,
        'psnr_per_iter': psnr_per_iter,
        'zs': zs,
        'gradient_time': gradient_time,
        'denoise_time': denoise_time
    }
    # return z, time_per_iter, psnr_per_iter, zs, gradient_time, denoise_time