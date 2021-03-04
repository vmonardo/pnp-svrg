from imports import *

class Problem():
    def __init__(self, img_path, img, H, W, lr_decay):
        if img_path is not None:
            original = np.array(Image.open(img_path).resize((H, W)))
        elif img is not None:
            original = img
        else:
            raise Exception('Need to pass in image path or image')

        original = (original - np.min(original)) / (np.max(original) - np.min(original))

        self.original = original
        self.H = H
        self.W = W
        self.lr_decay = lr_decay

    def batch(self, mini_batch_size):
        raise NotImplementedError('Need to implement batch() method')

    def full_grad(self, z):
        raise NotImplementedError('Need to implement full_grad() method')

    def stoch_grad(self, z, mini_batch_size):
        raise NotImplementedError('Need to implement stoch_grad() method')



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
        np.random.seed(0)
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

class Deblur(Problem):
    def __init__(self, img_path=None, img=None, H=64, W=64, 
                       kernel_path=None, kernel=None, sigma=1.0, subsampling=2, 
                       lr_decay=0.999):
        super().__init__(img_path, img, H, W, lr_decay)

        # problem setup
        if kernel_path is not None:
            blur = np.array(Image.open(kernel_path).resize((H, W)))
        elif kernel is not None:
            blur = kernel
        else:
            raise Exception('Need to pass in blur kernel path or kernel')
        
        N = H*W
        self.sigma = sigma
        self.subsampling = subsampling
        
        noises = np.random.normal(0, sigma, (N,))
        
        ## Create cirulant matrix
        vb = np.matrix.flatten(np.asarray(blur)) # flatten blurring kernel into a vector
        Cb = scipy.linalg.circulant(vb) # create circulant matrix of v_b

        ## Vectorize orig image
        vorig = np.matrix.flatten(np.asarray(self.original))
        
        ## Create noisy measurements
        y0 = Cb.dot(vorig) + noises
        y = y0[::subsampling]

        ## Precompute essential matrices
        idx = np.arange(0, H*W, subsampling)
        S_Cb = Cb[idx,:]
        
        ## Create initialization
        S_Cb_pinv = np.linalg.pinv(S_Cb)
        xinit = S_Cb_pinv.dot(y)
        
        self.num_meas = S_Cb.shape[0]
        self.noisy = xinit.reshape(H,W)
        self.y = y
        self.SCb = S_Cb
        
    def batch(self, mini_batch_size):
        N = self.num_meas
        tmp = np.random.permutation(N)
        k = tmp[0:mini_batch_size]
        batch = np.zeros(N)
        batch[k] = 1 
        return batch

    def full_grad(self, z):
        return (self.SCb.T.dot(self.SCb.dot(z.reshape(self.H*self.W,)) - self.y)).reshape(self.H,self.W)

    def stoch_grad(self, z, mini_batch_size):
        index = self.batch(mini_batch_size)
        res = self.SCb.dot(z.reshape(self.H*self.W,)) - self.y
        weights = np.multiply(index, res)
        return (self.SCb.T.dot(weights)).reshape(self.H,self.W)