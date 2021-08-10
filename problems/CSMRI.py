from problem import Problem
from imports import *

class CSMRI(Problem):
    def __init__(self, img_path=None, img=None, H=256, W=256, 
                       sample_prob=0.5, sigma=1.0,
                       lr_decay=0.999):
        super().__init__(img_path, img, H, W, lr_decay)

        self.sample_prob = sample_prob
        self.sigma = sigma

        # problem setup
        mask = np.random.choice([0, 1], size=(H, W), p=[1-sample_prob, sample_prob])

        forig = np.fft.fft2(self.original)
#         np.random.seed(0)
        noises = np.random.normal(0, sigma, (H, W))

        y0 = forig + noises
        y = np.multiply(mask, y0)
        x_init = np.absolute(np.fft.ifft2(y))

        i, j = np.meshgrid(np.arange(H), np.arange(W))
        omega = np.exp(-2*math.pi*1J/H)
        F = np.power(omega, i*j)

        self.noisy = x_init
        self.mask = mask
        self.y = y
        self.F = F

    def batch(self, mini_batch_size):
        # Get batch indices in terms of (row, col)

        H, W = self.mask.shape[:2]
        batch = np.zeros((1, H*W))
        tmp = np.linspace(0, H*W - 1, H*W)
        one_locs = tmp[np.matrix.flatten(self.mask) == 1].astype(int)
        batch_locs = np.random.choice(one_locs, mini_batch_size, replace=False)
        batch[0, batch_locs] = 1

        return batch.reshape(H, W).astype(int)

    def full_grad(self, z):
        N = self.H

        index = np.nonzero(self.mask)

        res = np.zeros((N, N), dtype=complex)
        
        F_i = self.F[index[0],:]
        F_j = self.F[index[1],:]
        
        res[index] = ((F_i @ z * F_j).sum(-1) - self.y[index])
        
        return (np.real(np.conj(self.F) @ res @ np.conj(self.F.T))/N**2)/len(index[0])

    def stoch_grad(self, z, mini_batch_size):
        N = self.H

        index = np.nonzero(self.batch(mini_batch_size))
        
        res = np.zeros((N, N), dtype=complex)
        
        F_i = self.F[index[0],:]
        F_j = self.F[index[1],:]
        
        res[index] = ((F_i @ z * F_j).sum(-1) - self.y[index])
        
        return (np.real(np.conj(self.F) @ res @ np.conj(self.F.T))/N**2)/len(index[0])