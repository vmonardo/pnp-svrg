#!/usr/bin/env python
# coding=utf-8
from problem import Problem
import numpy as np
import pylops
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

eps = 1e-10

class Deblur(Problem):
    def __init__(self, img_path=None, H=64, W=64, 
                       kernel_path=None, kernel=None, sigma=0.0, scale_percent=50):
        super().__init__(img_path, H, W)

        # User specified parameters
        self.sigma = sigma
        self.scale_percent = scale_percent

        # Identify kernel type
        if kernel_path is not None:
            self.kernel_path = kernel_path
        if kernel is not None:
            self.kernel = kernel
        if kernel_path is None and kernel is None:
            raise Exception('Need to pass in kernel path or kernel as image')

        self._load_kernel()

        # Calculate low-res image dimensions
        self.lrH = int(self.H * scale_percent / 100)
        self.lrW = int(self.W * scale_percent / 100)
        self.M = self.lrH*self.lrW

        self._generate_bop()
        
        # Blur the image with blurring kernel
        blurred = self.fft_blur(self.X, self.B)

        # create downsized, blurred image, as a vector
        y0 = self.Bop * blurred

        # create noise
        noises = np.random.normal(0, self.sigma, y0.shape)

        # add noise
        self.Y = y0 + noises

        # Initialize problem with least squares solution
        # solve using Pylops functionality
        D2op = pylops.Laplacian((self.H, self.W), weights=(1, 1), dtype='float64')

        xhat = pylops.optimization.leastsquares.NormalEquationsInversion(self.Bop, [D2op], self.Y,
                                                                 epsRs=[np.sqrt(0.01)],
                                                                 returninfo=False,
                                                                 **dict(maxiter=100))

        # Store initialization (as a vector)
        self.Xinit = self.fft_deblur(xhat, self.B)

    def _load_kernel(self):
        # Load the blurring kernel
        if self.kernel_path is not None:
            self.B = np.array(Image.open(self.kernel_path).resize((self.H, self.W)))
        elif self.kernel == "Identity":
            self.B = np.zeros(self.H*self.W)
            self.B[0] = 1
        elif self.kernel is not None:
            self.B = self.kernel
        else:
            raise Exception('Need to pass in blur kernel path or kernel')

    def _generate_bop(self):
        # Create bilinear interpolation operator using Pylops
        if self.scale_percent == 100:
            self.Bop = pylops.Identity(self.lrH*self.lrW)
        else: 
            # Create grid to instantiate downsized image
            ptsH = np.linspace(eps, self.H - (1 + eps), self.lrH)    
            ptsW = np.linspace(eps, self.W - (1 + eps), self.lrW)   
            meshH, meshZ = np.meshgrid(ptsH, ptsW)

            # create downsizing linear operator 
            iava = np.vstack([meshH.flatten(), meshZ.flatten()])
            self.Bop = pylops.signalprocessing.Bilinear(iava, (self.H, self.W))

    def forward_model(self, w):
        # Y = S B {x} (+ noise)
        return self.Bop*self.fft_blur(w, self.B)

    def f(self, w):
        # f(W) = 1 / 2*M || Y - M o F{W} ||_F^2
        # Compute data fidelity function value at a given point
        return np.linalg.norm(self.Y - self.forward_model(w)) ** 2 / 2 / self.M

    def fft_blur(self, M1, M2):
        return np.real(np.fft.ifft( np.fft.fft(M1.flatten())*np.fft.fft(M2.flatten()) )) 

    def fft_deblur(self, M1, M2):
        return np.real(np.fft.ifft( np.fft.fft(M1.flatten())/np.fft.fft(M2.flatten()) )) 

    ## nab l(x) = B^T S^T (S B Z - y) / m
    def grad_full(self, z):
        w = z.flatten()
        W_blurred = self.fft_blur(w, self.B)
        W_down = self.Bop * W_blurred
        res = W_down - self.Y
        res_up = self.Bop.H * res
        return self.fft_blur(res_up, np.roll(np.flip(self.B),1)) / self.M

    ## nab l(x) = B^T S^T (S B Z - y) / m
    def grad_stoch(self, z, mb):
        w = z.flatten()
        mb = mb.flatten()

        # Get nonzero indices of the mini-batch
        index = np.nonzero(mb)
        res = np.zeros(self.M)

        W_blurred = self.fft_blur(w, self.B)
        W_down = self.Bop * W_blurred
        res[index] = W_down[index] - self.Y[index]

        res_up = self.Bop.H * res
        return self.fft_blur(res_up, np.roll(np.flip(self.B),1))
    
# use this for debugging
if __name__ == '__main__':
    height = 64
    width = 64
    rescale = 50
    noise_level = 0.01

    p = Deblur(img_path='./data/Set12/01.png', kernel_path='./data/kernel.png', H=height, W=width, sigma=noise_level, scale_percent=rescale)
    p.grad_full_check()
    p.grad_stoch_check()
