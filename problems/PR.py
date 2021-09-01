#!/usr/bin/env python
# coding=utf-8
from PIL.Image import ENCODERS
from problem import Problem
import numpy as np
from numpy import linalg as la

class PhaseRetrieval(Problem):
    def __init__(self, img_path=None, H=256, W=256, 
                       num_meas=-1, sigma=1.0):
        super().__init__(img_path, H, W)

        # User specified parameters
        self.sigma = sigma
        self.M = num_meas

        # problem setup
        self.A = np.random.random((self.M,self.H*self.W)) + np.random.random((self.M,self.H*self.W)) * 1j
        tmp = self.forward_model(self.X)

        # create noise
        noises = np.random.normal(0, self.sigma, tmp.shape)

        self.Y = tmp + noises
        self.Xinit = self.spec_init()

    def spec_init(self):
        # create data matrix
        D = np.conj(self.A.T) @ np.diag(self.Y) @ self.A / 2 / self.M
        # find mean of measurements Y
        l = np.sum(self.Y) / self.M
        
        # run power method to find top eigenvector and eigenvalue
        tol = 1e-6
        m = 0
        mold = 1
        y_old = np.zeros(self.N)
        y_final = tol*np.ones(self.N) 
    
        while (np.abs(m - mold) > tol) and (la.norm(y_final - y_old) > tol):
            mold = m
            y_old = y_final
            tmp = np.dot(D, y_final)
            m = np.amax(tmp)
            y_final = tmp / m
        
        # print(np.abs(np.dot(self.X / la.norm(self.X), y_final / la.norm(y_final)) ** 2 ))
        return y_final * np.sqrt(m - l) 

    def forward_model(self, w):
        # Y = |Ax|
        return np.absolute(self.A.dot(w)) 

    def f(self, w):
        # f(W) = 1 / 2*M || y - F{W} ||_F^2
        # Compute data fidelity function value at a given point
        return np.linalg.norm(self.Y - self.forward_model(w)) ** 2 / 2 / self.M

    def grad_full(self, z):
        tmp = self.A @ z
        Weight = np.divide((np.absolute(tmp)  - self.Y),np.absolute(tmp))
        return np.real(np.conj(self.A).T @ np.diag(Weight) @ tmp ) / self.M 

    def grad_stoch(self, z, mb):
        index = np.nonzero(mb)
        A_gamma = self.A[index]
        tmp = A_gamma @ z
        Weight = np.divide((np.absolute(tmp)  - self.Y[index]),np.absolute(tmp))
        return np.real(np.conj(A_gamma).T @ np.diag(Weight) @ tmp) 

# use this for debugging
if __name__ == '__main__':
    height = 32
    width = 32
    alpha = 5       # ratio measurements / dimensions
    rescale = 50
    noise_level = 0.01

    p = PhaseRetrieval(img_path='./data/Set12/01.png', H=height, W=width, num_meas = alpha*height*width, sigma=noise_level)
    p.grad_full_check()
    p.grad_stoch_check()