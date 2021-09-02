#!/usr/bin/env python
# coding=utf-8
from PIL.Image import ENCODERS
from problem import Problem
import numpy as np
from numpy import linalg as la
from scipy.linalg import eigh

class PhaseRetrieval(Problem):
    def __init__(self, img_path=None, H=256, W=256, 
                       num_meas=-1, sigma=1.0):
        super().__init__(img_path, H, W)

        # User specified parameters
        self.sigma = sigma
        self.M = num_meas

        # problem setup
        self.A = np.random.randn(self.M,self.N) 

        Xnorm = la.norm(self.X)
        self.X0 = self.X / Xnorm # normalize

        y1 = self.forward_model(self.X0)

        # create noise
        # noises = np.random.normal(0, self.sigma, tmp.shape)

        # self.Y = tmp + noises
        self.Y = y1 # no noise
        tmp = self.spec_init()

        # Get sign of Xinit (solution is accurate up to a global phase shift)
        if tmp[tmp>0].shape[0] < tmp[tmp<0].shape[0]:
            self.Xinit = -tmp*Xnorm
        else:
            self.Xinit = tmp*Xnorm

    def spec_init(self):
        # create data matrix
        D = self.A.T.dot(self.A * self.Y[:,None]) / self.M
        D = (D + D.T) / 2
        w, v = eigh(D, eigvals=(self.N - 1,self.N - 1))
        v0 = -v/ la.norm(v) # why this gotta be negative tho man like idk
        return v0

    def forward_model(self, w):
        # Y = |Ax|
        return np.absolute(self.A.dot(w)) 

    def f(self, w):
        # f(W) = 1 / 2*M || y - F{W} ||_F^2
        # Compute data fidelity function value at a given point
        return np.linalg.norm(self.Y - self.forward_model(w)) ** 2 / 2 / self.M

    def grad_full(self, z):
        w = z.flatten()
        tmp = self.A @ w
        Weight = np.divide((np.absolute(tmp)  - self.Y),np.absolute(tmp))
        return np.real(np.conj(self.A).T.dot(Weight * tmp) ) / self.M 

    def grad_stoch(self, z, mb):
        index = np.nonzero(mb)
        w = z.flatten()
        A_gamma = self.A[index]
        tmp = A_gamma @ w
        Weight = np.divide((np.absolute(tmp)  - self.Y[index]),np.absolute(tmp))
        return np.real(np.conj(A_gamma).T.dot(Weight * tmp)) 

# use this for debugging
if __name__ == '__main__':
    height = 32
    width = 32
    alpha = 20       # ratio measurements / dimensions
    noise_level = 0

    p = PhaseRetrieval(img_path='./data/Set12/01.png', H=height, W=width, num_meas = alpha*height*width, sigma=noise_level)
    p.grad_full_check()
    p.grad_stoch_check()