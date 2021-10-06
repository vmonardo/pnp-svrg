#!/usr/bin/env python
# coding=utf-8
try:
    from .problem import Problem
except:
    from problem import Problem
import numpy as np
from numpy import linalg as la
from scipy.linalg import eigh
import time 

class PhaseRetrieval(Problem):
    def __init__(self, img_path=None, H=256, W=256, 
                       num_meas=-1, snr=None, sigma=None):
        super().__init__(img_path, H, W)
        
        # Name the problem
        self.pname = 'pr'

        # User specified parameters
        self.M = num_meas
        self.snr = snr
        self.sigma = sigma

        # problem setup
        self.A = np.random.randn(self.M,self.N)
        self.Y0 = self.forward_model(self.X).ravel()

        # Set noise details
        self.set_snr_sigma()

        # create noise
        noises = np.random.normal(0, self.sigma, self.Y0.shape)
        self.Y = self.Y0 + noises

        self.SNR = self.get_snr_from_sigma
        self.Xinit = np.random.uniform(0.0, 1.0, self.N) 
        # # self.Y = tmp + noises
        # tmp = self.spec_init()

        # # Get sign of Xinit (solution is accurate up to a global phase shift)
        # if tmp[tmp>0].shape[0] < tmp[tmp<0].shape[0]:
        #     self.Xinit = -tmp
        # else:
        #     self.Xinit = tmp

    def spec_init(self):
        # create data matrix
        D = self.A.T.dot(self.A * self.Y[:,None]) / self.M
        D = (D + D.T) / 2
        w, v = eigh(D, eigvals=(self.N - 1,self.N - 1))
        v0 = v/ la.norm(v) 
        return v0.ravel()

    def forward_model(self, w):
        # Y = |Ax|
        return np.absolute(self.A.dot(w)) 

    def f(self, w):
        # f(W) = 1 / 2*M || y - F{W} ||_F^2
        # Compute data fidelity function value at a given point
        return np.linalg.norm(self.Y - self.forward_model(w)) ** 2 / 2 / self.M

    def grad_full(self, z):
        w = z.ravel()
        tmp = self.A.dot(w).ravel()
        Weight = np.divide((np.absolute(tmp)  - self.Y.ravel()),np.absolute(tmp))
        return (np.conj(self.A).T.dot(Weight * tmp)).ravel() / self.M 

    def grad_stoch(self, z, mb):
        index = np.nonzero(mb)
        w = z.ravel()
        A_gamma = self.A[index]
        tmp = A_gamma.dot(w).ravel()
        Weight = np.divide((np.absolute(tmp)  - self.Y[index]),np.absolute(tmp))
        return (np.conj(A_gamma).T.dot(Weight * tmp)).ravel()

# use this for debugging
if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from denoisers import *
    from algorithms import *
    import timeit

    height = 128
    width = 128
    alpha = 1       # ratio measurements / dimensions
    noise_level = 0

    p = PhaseRetrieval(img_path='../data/Set12/01.png', H=height, W=width, num_meas = alpha*height*width, sigma=noise_level)
    mb = p.select_mb(100)
    print(128*128/100)
    print('1 full grads: ', timeit.timeit('p.grad_full(p.Xinit)', number=1, globals=globals()))
    print('1 stoch grads: ', timeit.timeit('p.grad_stoch(p.Xinit, mb)', number=1, globals=globals()))

    # p.grad_full_check()
    # p.grad_stoch_check()
    # p.Xinit = np.random.uniform(0.0, 1.0, p.N) # Try random initialization with the problem
    # print(p.Xinit.min(), p.Xinit.max())
    # print(p.Xinit)
    # print(np.dot(p.Xinit, p.X)**2/np.linalg.norm(p.Xinit)**2 / np.linalg.norm(p.X)**2)

    # denoiser = BM3DDenoiser()

    # # run for a while with super small learning rate and let hyperopt script find correct parameters :)
    # output_gd = pnp_gd(problem=p, denoiser=denoiser, eta=.2, tt=.1, verbose=True, converge_check=True, diverge_check=False)
    # time.sleep(1)
    # output_sgd = pnp_sgd(problem=p, denoiser=denoiser, eta=.001, tt=10, mini_batch_size=100, verbose=True, converge_check=False, diverge_check=False)
    # time.sleep(1)
    # output_sarah = pnp_sarah(problem=p, denoiser=denoiser, eta=.001, tt=10, T2=8, mini_batch_size=2, verbose=True, converge_check=False, diverge_check=False)
    # time.sleep(1)
    # output_saga = pnp_saga(problem=p, denoiser=denoiser, eta=.001, tt=10, mini_batch_size=2, hist_size=4, verbose=True, converge_check=False, diverge_check=False)
    # time.sleep(1)
    # output_svrg = pnp_svrg(problem=p, denoiser=denoiser, eta=.002, tt=10, T2=8, mini_batch_size=2, verbose=True, converge_check=False, diverge_check=False)