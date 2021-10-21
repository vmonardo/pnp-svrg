import torch
import matplotlib.pyplot as plt

# import modules
from denoisers import *
from problems import *
from algorithms import *

# create problem 
im_height, im_width = 256, 256  # Image dimensions
samp_rate = 0.7                 # Pick a number 0 < SR <= 1
snr = 10.
main_problem = CSMRI('./data/13.png', H=im_height, W=im_width, sample_prob=samp_rate, snr=snr)

# load the model
CNNDenoiser = RealSN_DnCNNDenoiser(model_type="RealSN_DnCNN", sigma=5)

with torch.no_grad():
    results_svrg = pnp_svrg(main_problem, denoiser=CNNDenoiser, eta=1e-1, tt=10, T2=10, mini_batch_size=500, verbose=True)

    plt.figure()
    plt.subplot(311)
    plt.imshow(main_problem.Xrec, cmap='gray')
    plt.subplot(312)
    plt.imshow(main_problem.Xinit.reshape(im_height, im_width), cmap='gray')
    plt.subplot(313)
    plt.imshow(results_svrg['z'].reshape(im_height, im_width), cmap='gray')
    plt.show()



