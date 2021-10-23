#!/usr/bin/env python
# coding=utf-8

try:
    from .problem import Problem
except:
    from problem import Problem
import numpy as np
import math

class CSMRI(Problem):
    def __init__(self, img_path=None, H=256, W=256, 
                       sample_prob=0.5, snr=None, sigma=None):
        super().__init__(img_path, H, W)

        # Name the problem
        self.pname = 'csmri'

        # User specified parameters
        self.sample_prob = sample_prob
        self.snr = snr
        self.sigma = sigma
        
        self._generate_mask()
        self._generate_F()

        self.Y0 = self.forward_model(self.X)

        # Set noise details
        self.set_snr_sigma()

        noises = np.random.normal(0, self.sigma, self.Y0.shape)
        self.Y = self.Y0 + np.multiply(self.mask, noises)
        self.SNR = self.get_snr_from_sigma
        self.Xinit = np.absolute(np.fft.ifft2(self.Y)).ravel()
        self.Xinit = (self.Xinit - np.min(self.Xinit)) / (np.max(self.Xinit) - np.min(self.Xinit))
        
        # maintaining consistency for debugging
        self.lrH, self.lrW = self.H, self.W   
        self.M = self.N                         # dimensions of actual output signal  
        self.M0 = np.count_nonzero(self.mask)   # number of measurements

    def _generate_mask(self):
        # Generate random binary mask to determine sampled Fourier coefficients
        self.mask = np.random.choice([0, 1], size=(self.H, self.W), p=[1-self.sample_prob, self.sample_prob])

    def _generate_F(self):
        # Get 2D Fourier Transform Matrix
        i, j = np.meshgrid(np.arange(self.H), np.arange(self.W))
        omega = np.exp(-2*math.pi*1J/self.H)
        self.F = np.power(omega, i*j)

    def forward_model(self, w):
        # Forward Model: Y = M o (F{X} + noise)
        # return as a vector
        tmp = w.reshape(self.H, self.W)
        # ftmp = np.fft.fft2(tmp)
        ftmp = self.F.dot(tmp).dot(self.F.T)      
        return np.multiply(self.mask, ftmp)

    def f(self, w):
        # f(W) = || Y - M o F{W} ||_F^2 / 2*M
        # Compute data fidelity function value at a given point
        return np.linalg.norm(self.Y - self.forward_model(w)) ** 2 / 2 / self.M

    def select_mb(self, size):
        # Draw measurements uniformly at random for mini-batch stochastic gradient
        if size > self.M:
            print('MB size is too big: ', size, ' > ', self.M)
        batch = np.zeros(self.M)
        mask_locs = np.asarray(np.flatnonzero(self.mask))
        batch_locs = np.random.choice(mask_locs, size, replace=False)
        batch[batch_locs] = 1
        return batch.reshape(self.H, self.W).astype(int)

    def grad_full(self, z):
        tmp = np.fft.fft2(z.reshape(self.H, self.W))
        res = tmp * self.mask
        index = np.nonzero(self.mask)
        res[index] = res[index] - self.Y[index]
        return np.real(np.fft.ifft2(res)).ravel() / self.M0

    def grad_stoch(self, z, mb):
        mbb = self.mask * mb
        tmp = np.fft.fft2(z.reshape(self.H, self.W))
        res = tmp * mbb
        index = np.nonzero(mbb)
        res[index] = res[index] - self.Y[index]
        return np.real(np.fft.ifft2(res)).ravel()

# use this for debugging
if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from denoisers import *
    from algorithms import *
    import time
    import timeit

    height = 128
    width = 128 
    noise_level = 0.0

    # create "ideal" problem
    p = CSMRI(img_path='../data/Set12/01.png', H=height, W=width, sample_prob=1., snr=10)
    # p.grad_full_check()
    # p.grad_stoch_check()
    p.Xinit = np.random.uniform(0.0, 1.0, p.N) # Try random initialization with the problem
    print(p.snr, p.sigma)

    mb1 = p.select_mb(1)
    mb2 = np.count_nonzero(p.mask)
    # index = np.nonzero(mb1)

    print('100 full grads: ', timeit.timeit('p.grad_full(p.Xinit)', number=100, globals=globals()))
    print('100 stoch grads: ', timeit.timeit('p.grad_stoch(p.Xinit, mb1)', number=100, globals=globals()))

    # run for a while with super small learning rate and let hyperopt script find correct parameters :)
    # output_gd = pnp_gd(problem=p, denoiser=denoiser, eta=.2, tt=.1, verbose=True, converge_check=True, diverge_check=False)
    # time.sleep(1)
    # output_sgd = pnp_sgd(problem=p, denoiser=denoiser, eta=.001, tt=10, mini_batch_size=1, verbose=True, converge_check=False, diverge_check=False)
    # time.sleep(1)
    # output_sarah = pnp_sarah(problem=p, denoiser=denoiser, eta=.001, tt=.1, T2=8, mini_batch_size=2, verbose=True, converge_check=False, diverge_check=False)
    # time.sleep(1)
    # output_saga = pnp_saga(problem=p, denoiser=denoiser, eta=.001, tt=10, mini_batch_size=2, hist_size=16, verbose=True, converge_check=False, diverge_check=False)
    # time.sleep(1)
    # output_svrg = pnp_svrg(problem=p, denoiser=denoiser, eta=.002, tt=10, T2=8, mini_batch_size=2, verbose=True, converge_check=False, diverge_check=False)