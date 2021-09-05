#!/usr/bin/env python
# coding=utf-8

from numpy.core.fromnumeric import reshape
from problem import Problem
import numpy as np
import math
import time

class CSMRI(Problem):
    def __init__(self, img_path=None, H=256, W=256, 
                       sample_prob=0.5, sigma=1.0):
        super().__init__(img_path, H, W)

        # Name the problem
        self.pname = 'csmri'

        # User specified parameters
        self.sample_prob = sample_prob
        self.sigma = sigma
        self._generate_mask()
        self._generate_F()

        y0 = self.forward_model(self.X)
        noises = np.random.normal(0, self.sigma, y0.shape)
        self.Y = y0 + np.multiply(self.mask, noises)
        self.Xinit = np.absolute(np.fft.ifft2(self.Y))

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

    def grad_full(self, z):
        # initialize space for residual and compute it
        res = np.zeros(z.shape, dtype=complex)
        res = (self.forward_model(z) - self.Y)

        # return inverse 2D Fourier Transform of residual
        # tmp = np.real(np.fft.ifft2(res))
        return np.real(np.conj(self.F).dot(res).dot(np.conj(self.F.T))).ravel() / self.M

    def grad_stoch(self, z, mb):
        # Get objects as images
        w = z.reshape(self.H,self.W)
        mb = np.multiply(mb.reshape(self.H, self.W), self.mask)

        # Get nonzero indices of the mini-batch
        index = np.nonzero(mb)

        # Get relevant rows of DFT matrix
        F_i = self.F[index[0],:]
        F_j = self.F[index[1],:]

        # compute residual
        res = ((F_i @ w * F_j).sum(-1) - self.Y[index])  # get residual in as a column vector
        tmp = np.einsum('ij,i->ij',np.conj(F_i),res)

        # return inverse 2D Fourier Transform of residual
        return np.real(np.einsum('ij,ik->jk',tmp,np.conj(F_j))).ravel()

# use this for debugging
if __name__ == '__main__':
    height = 32
    width = 32
    noise_level = 0.0

    # create "ideal" problem
    p = CSMRI(img_path='./data/Set12/01.png', H=height, W=width, sample_prob=1, sigma=noise_level)
    p.grad_full_check()
    p.grad_stoch_check()
    p.Xinit = np.random.uniform(0.0, 1.0, p.N) # Try random initialization with the problem
    import sys
    sys.path.append('denoisers/')
    from NLM import NLMDenoiser
    denoiser = NLMDenoiser(sigma_est=0, patch_size=4, patch_distance=5)
    sys.path.append('algorithms/')
    from pnp_gd import pnp_gd
    from pnp_sgd import pnp_sgd
    from pnp_sarah import pnp_sarah
    from pnp_saga import pnp_saga
    from pnp_svrg import pnp_svrg

    # run for a while with super small learning rate and let hyperopt script find correct parameters :)
    output_gd = pnp_gd(problem=p, denoiser=denoiser, eta=.2, tt=.1, verbose=True, converge_check=True, diverge_check=False)
    time.sleep(1)
    output_sgd = pnp_sgd(problem=p, denoiser=denoiser, eta=.001, tt=10, mini_batch_size=1, verbose=True, converge_check=False, diverge_check=False)
    time.sleep(1)
    output_sarah = pnp_sarah(problem=p, denoiser=denoiser, eta=.001, tt=10, T2=8, mini_batch_size=2, verbose=True, converge_check=False, diverge_check=False)
    time.sleep(1)
    output_saga = pnp_saga(problem=p, denoiser=denoiser, eta=.001, tt=10, mini_batch_size=2, hist_size=16, verbose=True, converge_check=False, diverge_check=False)
    time.sleep(1)
    output_svrg = pnp_svrg(problem=p, denoiser=denoiser, eta=.002, tt=10, T2=8, mini_batch_size=2, verbose=True, converge_check=False, diverge_check=False)