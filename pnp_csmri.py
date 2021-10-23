import torch
import matplotlib.pyplot as plt
import numpy as np

# import modules
from denoisers import *
from denoisers.MMODenoise import MMODenoiser
from problems import *
from algorithms import *

# create problem 
im_height, im_width = 256, 256  # Image dimensions
samp_rate = 0.5                 # Pick a number 0 < SR <= 1
snr = 30.
main_problem = CSMRI('./data/13.png', H=im_height, W=im_width, sample_prob=samp_rate, snr=snr)

# load denoisers
BM3DDenoiser = BM3DDenoiser()
NLMDenoiser = NLMDenoiser()
TVDenoiser = TVDenoiser()

# compare algorithms with RealSN_DnCNN Denoiser
with torch.no_grad():
    CNNDenoiser = RealSN_DnCNNDenoiser(model_type="RealSN_DnCNN", sigma=5)
    tmp = CNNDenoiser.denoise(main_problem.Xrec)
    results_gd = pnp_gd(main_problem, denoiser=CNNDenoiser, eta=1e4, tt=10, verbose=True, converge_check=False, diverge_check=False)
    results_sgd = pnp_sgd(main_problem, denoiser=CNNDenoiser, eta=1e4, tt=10, mini_batch_size=main_problem.M0, verbose=True, converge_check=False, diverge_check=False)
    results_svrg = pnp_svrg(main_problem, denoiser=CNNDenoiser, eta=1e4, tt=10, T2=1, mini_batch_size=main_problem.M0, verbose=True, converge_check=False, diverge_check=False)

    # display output
    plt.figure(figsize=(20,4))
    plt.subplot(151)
    plt.imshow(main_problem.Xrec, cmap='gray')
    plt.title('Original')
    plt.subplot(152)
    Xinit = main_problem.Xinit.reshape(im_height, im_width)
    plt.imshow(Xinit, cmap='gray')
    plt.title('Initial, PSNR: ' + str(main_problem.PSNR(Xinit)))
    plt.subplot(153)
    plt.imshow(results_gd['z'].reshape(im_height, im_width), cmap='gray')
    plt.title(results_gd['algo_name'] + ', PSNR: ' + str(main_problem.PSNR(results_gd['z'])))
    plt.subplot(154)
    plt.imshow(results_sgd['z'].reshape(im_height, im_width), cmap='gray')
    plt.title(results_sgd['algo_name'] + ', PSNR: ' +  str(main_problem.PSNR(results_sgd['z'])))
    plt.subplot(155)
    plt.imshow(results_svrg['z'].reshape(im_height, im_width), cmap='gray')
    plt.title(results_svrg['algo_name'] + ', PSNR: ' +  str(main_problem.PSNR(results_svrg['z'])))

    psnr_fig = plt.figure(figsize=(6,6))
    psnr_ax = psnr_fig.add_subplot(1,1,1)
        
    tArray1 = results_svrg['time_per_iter']
    psnrArray1 = results_svrg['psnr_per_iter']
    tArray2 = results_sgd['time_per_iter']
    psnrArray2 = results_sgd['psnr_per_iter']
    tArray3 = results_gd['time_per_iter']
    psnrArray3 = results_gd['psnr_per_iter']
    print(len(tArray1), len(tArray2), len(tArray3))
    psnr_ax.plot(np.cumsum(tArray1), psnrArray1, "b", linewidth=3, label=str(results_svrg['algo_name']))
    psnr_ax.plot(np.cumsum(tArray1)[::30], psnrArray1[::30], "b*", markersize=10)
    psnr_ax.plot(np.cumsum(tArray2), psnrArray2, "r", linewidth=3, label=str(results_sgd['algo_name']))
    psnr_ax.plot(np.cumsum(tArray2)[::30], psnrArray2[::30], "r*", markersize=10)
    psnr_ax.plot(np.cumsum(tArray3), psnrArray3, "k", linewidth=3, label=str(results_gd['algo_name']))
    psnr_ax.plot(np.cumsum(tArray3)[::30], psnrArray3[::30], "k*", markersize=10)
    psnr_ax.set(xlabel='time (s)', ylabel='PSNR (dB)')
    psnr_ax.legend()
    psnr_ax.grid()
    psnr_fig.tight_layout()
    plt.show()

    # compare denoisers with PnP-SVRG

