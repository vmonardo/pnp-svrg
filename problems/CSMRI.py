#!/usr/bin/env python
# coding=utf-8

from problem import Problem
import numpy as np
import math

class CSMRI(Problem):
    def __init__(self, img_path=None, H=256, W=256, 
                       sample_prob=0.5, sigma=1.0):
        super().__init__(img_path, H, W)

        # User specified parameters
        self.sample_prob = sample_prob
        self.sigma = sigma

        self._generate_problem()

    def _generate_problem(self):
        # Problem Setup
        self._generate_mask()
        self._generate_F()

        y0 = self.forward_model(self.X)
        noises = np.random.normal(0, self.sigma, (self.H, self.W))
        y = y0.reshape(self.H, self.W) + np.multiply(self.mask, noises)
        x_init = np.absolute(np.fft.ifft2(y))

        # Save essential variables
        self.Xinit = x_init.reshape(self.N)
        self.Y = y.flatten()

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
        ftmp = np.fft.fft2(tmp)      
        return np.multiply(self.mask, ftmp).reshape(self.N)

    def f(self, w):
        # f(W) = 1 / 2*M || Y - M o F{W} ||_F^2
        # Compute data fidelity function value at a given point
        return np.linalg.norm(self.Y - self.forward_model(w)) ** 2 / 2 / self.H / self.W

    def grad_full(self, z):
        # Get objects as images
        y = self.Y.reshape(self.H, self.W)
        w = z.reshape(self.H, self.W)

        # initialize space for residual and compute it
        res = np.zeros((self.H, self.W), dtype=complex)
        res = (self.forward_model(w).reshape(self.H,self.W) - y)

        # return inverse 2D Fourier Transform of residual
        tmp = np.real(np.fft.ifft2(res))
        return tmp.reshape(self.N) * 2 

    def grad_stoch(self, z, mb):
        # Get objects as images
        y = self.Y.reshape(self.H, self.W)
        w = z.reshape(self.H, self.W)

        # index = np.nonzero(mb)
        # initialize space for residual and compute it
        res = np.zeros((self.H, self.W), dtype=complex)
        res = np.multiply((self.forward_model(w).reshape(self.H,self.W) - y), mb)
        
        # return inverse 2D Fourier Transform of residual
        tmp = np.real(np.fft.ifft2(res))
        return tmp.reshape(self.N) * 2 

# use this for debugging
if __name__ == '__main__':
    height = 64
    width = 64
    noise_level = 0.01

    p = CSMRI(img_path='./data/Set12/01.png', H=height, W=width, sample_prob=0.5, sigma=noise_level)
    p.grad_full_check()
    p.grad_stoch_check()
