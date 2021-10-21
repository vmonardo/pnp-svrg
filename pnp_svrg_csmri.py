"""
    Plug and Play FBS for Compressive Sensing MRI
    Authors: Jialin Liu (danny19921123@gmail.com)
"""

import os
import numpy as np
from PIL import Image
import torch
import cv2
import argparse
import glob
import scipy.io as sio
import scipy.misc
from denoisers.DeepDenoisers.utils.utils import load_model
from denoisers.DeepDenoisers.utils.utils import psnr
from denoisers.DeepDenoisers.utils.config import analyze_parse
import matplotlib.pyplot as plt

def get_batch(mask, size, m, n):
    # Draw measurements uniformly at random for mini-batch stochastic gradient
    if size > m*n:
        print('MB size is too big: ', size, ' > ', m*n)
    batch = np.zeros(m*n)
    mask_locs = np.asarray(np.flatnonzero(mask))
    batch_locs = np.random.choice(mask_locs, size, replace=False)
    batch[batch_locs] = 1
    return batch.reshape(m, n).astype(int)

def pnp_fbs_csmri(denoise_func, im_orig, mask, noises, **opts):

    alpha    = opts.get('alpha', 0.4)
    maxitr1 = opts.get('maxitr1', 100)
    maxitr2 = opts.get('maxitr2', 10)
    mb_size = opts.get('mb_size', 100)
    verbose = opts.get('verbose',1)
    sigma = opts.get('sigma', 5)

    """ Initialization. """

    m, n = im_orig.shape
    index = np.nonzero(mask)

    y = np.fft.fft2(im_orig) * mask + noises # observed value
    x_init = np.fft.ifft2(y) # zero fill
    x_init = np.abs(x_init)

    print(psnr(x_init,im_orig))

    x = np.copy(np.abs(x_init))

    """ Main loop. """
    for i in range(maxitr1):
        res = np.fft.fft2(x) * mask
        index = np.nonzero(mask)
        res[index] = res[index] - y[index]
        mu = np.fft.ifft2(res)

        w = np.copy(x)

        for j in range(maxitr2):
            # create mini-batch
            BATCH = get_batch(mask, mb_size, m, n)
            tmp = mask * BATCH
            
            """ Update variables. """
            # get stochastic gradient of reference point
            res = np.fft.fft2(w) * tmp
            index = np.nonzero(tmp)
            res[index] = res[index] - y[index]
            w_grad = np.fft.ifft2(res)
            
            # get stochastic gradient of current iterate
            res = np.fft.fft2(x) * tmp
            index = np.nonzero(tmp)
            res[index] = res[index] - y[index]
            x_grad = np.fft.ifft2(res)

            # compute SVRG gradient update
            v = (x_grad - w_grad) / mb_size + mu

            # gradient update
            x = x - alpha * v
            x = np.abs( x )

            """ Monitoring. """

            # psnr
            if verbose:
                print("i: {}, j: {}, \t psnr: {}"\
                    .format(i+1, j+1, psnr(x,im_orig)))

            xout = np.copy(x)


            """ Denoising step. """

            xtilde = np.copy(x)
            mintmp = np.min(xtilde)
            maxtmp = np.max(xtilde)
            xtilde = (xtilde - mintmp) / (maxtmp - mintmp)
            
            # the reason for the following scaling:
            # our denoisers are trained with "normalized images + noise"
            # so the scale should be 1 + O(sigma)
            scale_range = 1.0 + sigma/255.0/2.0
            scale_shift = (1 - scale_range) / 2.0
            xtilde = xtilde * scale_range + scale_shift

            # pytorch denoising model
            xtilde_torch = np.reshape(xtilde, (1,1,m,n))
            xtilde_torch = torch.from_numpy(xtilde_torch).type(torch.FloatTensor).cuda()
            r = denoise_func(xtilde_torch).cpu().numpy()
            r = np.reshape(r, (m,n))
            x = xtilde - r

            # scale and shift the denoised image back
            x = (x - scale_shift) / scale_range
            x = x * (maxtmp - mintmp) + mintmp


    return xout

# ---- input arguments ----
opt = analyze_parse(5, 0.1, 10, 20, 100) # the arguments are default sigma, default alpha and default max iteration.

# ---- load the model ---- 
model = load_model(opt.model_type, opt.sigma)

with torch.no_grad():

    # ---- load the ground truth ----
    im_orig = cv2.imread('denoisers/DeepDenoisers/Demo_mat/CS_MRI/Brain.jpg', 0)/255.0

    # ---- load mask matrix ----
    mat = sio.loadmat('denoisers/DeepDenoisers/Demo_mat/CS_MRI/Q_Random30.mat')
    mask = mat.get('Q1').astype(np.float64)

    # ---- load noises -----
    noises = sio.loadmat('denoisers/DeepDenoisers/Demo_mat/CS_MRI/noises.mat')
    noises = noises.get('noises').astype(np.complex128) * 5.0

    # ---- set options -----
    opts = dict(sigma = opt.sigma, alpha=opt.alpha, maxitr1=opt.maxitr1, verbose=opt.verbose) 

    # ---- plug and play !!! -----
    out = pnp_fbs_csmri(model, im_orig, mask, noises, **opts)

    # ---- print result ----- 
    print('Plug-and-Play PNSR: ', psnr(out,im_orig))

    y = np.fft.fft2(im_orig) * mask + noises # observed value
    x_init = np.fft.ifft2(y) # zero fill

    plt.figure()
    plt.subplot(311)
    plt.imshow(im_orig, cmap='gray')
    plt.subplot(312)
    plt.imshow(np.abs(x_init), cmap='gray')
    plt.subplot(313)
    plt.imshow(out, cmap='gray')
    plt.show()



