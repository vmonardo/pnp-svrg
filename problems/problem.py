#!/usr/bin/env python
# coding=utf-8
from PIL import Image
import numpy as np
from numpy.lib.npyio import save
from skimage.metrics import peak_signal_noise_ratio

class Problem():
    def __init__(self, img_path, H, W):
        # User specified parameters
        self.H = H                  # Height of the image
        self.W = W                  # Width of the image
        self.N = H*W                # Dimensionality of the problem
        self.M = self.N             # BE SURE TO SET IN YOUR PROBLEM

        # Load in Image to specified dimensions (H,W)
        if img_path is not None:
            tmp = np.array(Image.open(img_path).resize((H, W)))
        else:
            raise Exception('Need to pass in image path or image')

        # Normalize image such that all pixels are in rage [0,1]
        tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))
        self.Xrec = tmp         # As image
        self.X = tmp.ravel()    # As vector

        # Initialize essential parameters
        # self.Y = np.empty(self.M)
        self.Xinit = np.empty_like(self.X)
    
    def get_item(self, key):
        return self.__dict__[key]

    def PSNR(self, w):
        # return PSNR w.r.t. ground truth image
        return peak_signal_noise_ratio(self.Xrec, w.reshape(self.H, self.W))

    def display(self, color_map='gray', show_measurements=False, save_results=False, save_dir='figures/', show_figs=False):
        self.color_map = color_map
        import matplotlib.pyplot as plt
        if save_results:
            from datetime import datetime
            import os
            baseFileName = save_dir + self.pname + '/' + datetime.now().strftime('%y-%m-%d-%H-%M') + "/"
            self.prob_dir = baseFileName    # Set directory to say results
            os.makedirs(baseFileName, exist_ok=True)

        # Display original image
        orig_fig = plt.figure(figsize=(3,3))
        plt.imshow(self.Xrec, cmap=color_map, vmin=0, vmax=1)
        plt.title('Original Image')
        plt.xticks([])
        plt.yticks([])
        if save_results:
            fileName = baseFileName + 'original.eps'
            orig_fig.savefig(fileName, transparent = True, bbox_inches = 'tight', pad_inches = 0)
        if show_figs:
            plt.show()

        # Display initialization
        init_fig = plt.figure(figsize=(3,3))
        plt.imshow(self.Xinit.reshape(self.H, self.W), cmap=color_map, vmin=0, vmax=1)
        plt.title('Initialization')
        plt.xticks([])
        plt.yticks([])
        if save_results:
            fileName = baseFileName + 'initialization.eps'
            init_fig.savefig(fileName, transparent = True, bbox_inches = 'tight', pad_inches = 0)
        if show_figs:
            plt.show()

        if show_measurements:
            meas_fig = plt.figure(figsize=(3,3))
            plt.imshow(self.Y.reshape(self.lrH, self.lrW), cmap=color_map, vmin=0, vmax=1)
            plt.title('Measurements')
            plt.xticks([])
            plt.yticks([])
            if save_results:
                fileName = baseFileName + 'measurements.eps'
                meas_fig.savefig(fileName, transparent = True, bbox_inches = 'tight', pad_inches = 0)
            if show_figs:
                plt.show()

    def select_mb(self, size):
        # Draw measurements uniformly at random for mini-batch stochastic gradient
        batch = np.zeros(self.M)
        batch_locs = np.random.choice(self.M, size, replace=False)
        batch[batch_locs] = 1
        return batch.astype(int)

    def f(self, z):
        # Method to compute the data fidelity loss at a given input
        raise NotImplementedError('Need to implement f() method')

    def grad_full(self, z):
        # Compute a full gradient w.r.t. data fidelity term
        raise NotImplementedError('Need to implement full_grad() method')

    def grad_stoch(self, z, mb_indices):
        # Compute a stochastic gradient w.r.t. data fidelity term at mini-batch indices given
        raise NotImplementedError('Need to implement stoch_grad() method')

    def grad_full_check(self):
        # Check the gradient implementation at a random value
        w = np.random.uniform(0.0, 1.0, self.N)
        
        delta = np.zeros(self.N)
        grad = np.zeros(self.N)
        eps = 1e-6

        for i in range(self.N):
            delta[i] = eps
            grad[i] = (self.f(w+delta) -  self.f(w)) / eps   
            delta[i] = 0

        grad_comp = self.grad_full(w).ravel()
        print('grad: ', grad)
        print('grad_comp: ', grad_comp)
        if np.linalg.norm(grad - grad_comp) > 1e-4:
            print('Full Grad check failed!')
            print('norm: ', np.linalg.norm(grad - grad_comp))
            print('grad diff: ', grad)
            print('grad compute: ', grad_comp)
            return False
        else:
            print('Full Grad check succeeded!')
            return True
        
    def grad_stoch_check(self):
        # check that (grad) f(w) = sum((grad) f_i(w)) / m at a random value
        w = np.random.uniform(0.0, 1.0, self.N)
        full_grad = self.grad_full(w)
        grad_comp = np.zeros(self.N)

        for i in range(self.M):
            mb = np.zeros(self.M, dtype=int)
            mb[i] = 1
            grad_comp += self.grad_stoch(w, mb)

        if np.linalg.norm(full_grad.ravel() - grad_comp / self.M) > 1e-6:
            print('Stoch Grad check failed!')
            print('full grad: ', full_grad)
            print('grad comp: ', grad_comp)
            return False
        else:
            print('Stoch Grad check succeeded!')
            return True

# use this for debugging
if __name__ == '__main__':
    height = 64
    width = 64
    noise_level = 0.01

    p = Problem(img_path='./data/Set12/01.png', H=height, W=width)
    x = p.select_mb(height*width)
    print(x, x.shape, p.N)