#!/usr/bin/env python
# coding=utf-8
from problem import Problem
import numpy as np
import pylops
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import time

eps = 1e-10

class Deblur(Problem):
    def __init__(self, img_path=None, H=64, W=64, 
                       kernel_path=None, kernel=None, sigma=0.0, scale_percent=50):
        super().__init__(img_path, H, W)

        # Name the problem
        self.pname = 'deblur'

        # User specified parameters
        self.sigma = sigma
        self.scale_percent = scale_percent

        # Identify kernel type
        self.kernel_path = kernel_path
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
        # create downsized, blurred image, as a vector
        y0 = self.forward_model(self.X)

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
            # no blurring
            self.B = np.zeros(self.N)
            self.B[0] = 1
        elif self.kernel == "Minimal":
            # small blur
            self.B = np.zeros((self.H, self.W))
            self.B[0,0] = 1
            self.B[self.H//2,self.H//2] = 1
            self.B[self.H//2,self.H//3] = 1
            self.B[self.H//2,self.H//4] = 1
            self.B /= 4
        elif self.kernel is not None:
            self.B = self.kernel
        else:
            raise Exception('Need to pass in blur kernel path or kernel')
        # storing everything as arrays for consistency
        self.B = self.B.ravel() / self.N

    def _generate_bop(self):
        # Create bilinear interpolation operator using Pylops
        if self.scale_percent == 100:
            self.Bop = pylops.Identity(self.lrH*self.lrW)
        else: 
            # Create grid to instantiate downsized image
            ptsH = np.linspace(eps, self.H - (1 + eps), self.lrH)    
            ptsW = np.linspace(eps, self.W - (1 + eps), self.lrW)   
            meshW, meshH = np.meshgrid(ptsH, ptsW)                  # idk why W and H have to be flipped
                                                                    # but result is transposed if not 

            # create downsizing linear operator 
            iava = np.vstack([meshH.ravel(), meshW.ravel()])
            self.Bop = pylops.signalprocessing.Bilinear(iava, (self.H, self.W))

    def forward_model(self, w):
        # Y = S B {x} (+ noise)
        return self.Bop*self.fft_blur(w, self.B)

    def f(self, w):
        # f(W) = 1 / 2*M || Y - S B{W} ||_F^2
        # Compute data fidelity function value at a given point
        return np.linalg.norm(self.Y - self.forward_model(w)) ** 2 / 2 / self.M

    def fft_blur(self, M1, M2):
        return np.real(np.fft.ifft( np.fft.fft(M1.ravel())*np.fft.fft(M2.ravel()) ))
         
    def fft_deblur(self, M1, M2):
        return np.real(np.fft.ifft( np.fft.fft(M1.ravel())/np.fft.fft(M2.ravel()) ))

    ## nab l(x) = B^T S^T (S B Z - y) / m
    def grad_full(self, z):
        w = z.ravel()
        W_blurred = self.fft_blur(w, self.B)
        W_down = self.Bop * W_blurred
        res = W_down - self.Y
        res_up = self.Bop.H * res
        return self.fft_blur(res_up, np.roll(np.flip(self.B),1)) / self.M

    ## nab l(x) = B^T S^T (S B Z - y) / m
    def grad_stoch(self, z, mb):
        w = z.ravel()
        mb = mb.ravel()

        # Get nonzero indices of the mini-batch
        index = np.nonzero(mb)
        res = np.zeros(self.M,)
        W_blurred = self.fft_blur(w, self.B)
        W_down = self.Bop * W_blurred
        res[index] = W_down[index] - self.Y[index]

        res_up = self.Bop.H * res
        return self.fft_blur(res_up, np.roll(np.flip(self.B),1))
    
# use this for debugging
if __name__ == '__main__':
    height = 4
    width = 4
    rescale = 100
    noise_level = 0.0

    # Create "ideal" problem
    # p = Deblur(img_path='./data/Set12/01.png', kernel_path='./data/kernel.png', H=height, W=width, sigma=noise_level, scale_percent=rescale)
    p = Deblur(img_path='./data/Set12/01.png', kernel_path='./data/kernel.png', H=height, W=width, sigma=noise_level, scale_percent=rescale)
    p.grad_full_check()
    p.grad_stoch_check()
    print(p.B)
    time.sleep(1)
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
    output_gd = pnp_gd(problem=p, denoiser=denoiser, eta=1, tt=20, verbose=True, converge_check=False, diverge_check=True)
    time.sleep(1)
    output_sgd = pnp_sgd(problem=p, denoiser=denoiser, eta=1, tt=20, mini_batch_size=1, verbose=True, converge_check=False, diverge_check=True)
    time.sleep(1)
    output_sarah = pnp_sarah(problem=p, denoiser=denoiser, eta=1, tt=20, T2=8, mini_batch_size=1, verbose=True, converge_check=False, diverge_check=True)
    time.sleep(1)
    output_saga = pnp_saga(problem=p, denoiser=denoiser, eta=1, tt=20, mini_batch_size=1, hist_size=16, verbose=True, converge_check=False, diverge_check=True)
    time.sleep(1)
    output_svrg = pnp_svrg(problem=p, denoiser=denoiser, eta=1, tt=20, T2=8, mini_batch_size=1, verbose=True, converge_check=False, diverge_check=True)

    print(output_gd['psnr_per_iter'][-1],output_gd['psnr_per_iter'][-1], output_sgd['psnr_per_iter'][-1], output_sarah['psnr_per_iter'][-1], output_saga['psnr_per_iter'][-1], output_svrg['psnr_per_iter'][-1])