from imports import *
import pylops

class Problem():
    def __init__(self, img_path, img, H, W, lr_decay=1):
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

class Deblur(Problem):
    def __init__(self, img_path=None, img=None, H=64, W=64, 
                       kernel_path=None, kernel=None, sigma=0.0, scale_percent=50, 
                       lr_decay=0.999):
        super().__init__(img_path, img, H, W, lr_decay)

        # problem setup
        if kernel_path is not None:
            blur = np.array(Image.open(kernel_path).resize((H, W)))
        elif kernel == "Identity":
            blur = np.zeros(H*W)
            blur[0] = 1
        elif kernel is not None:
            blur = kernel
        else:
            raise Exception('Need to pass in blur kernel path or kernel')
        
        self.sigma = sigma
        self.blur = blur
        img = self.original
        nz, nx = img.shape
        self.dim_old = (nz, nx)
        
        # Blur the image with blurring kernel
        blurred = fft_blur(img, blur)

        if scale_percent == 100:
            width, height = nz, nx
            self.dim_new = self.dim_old
            Bop = pylops.Identity(width*height)
        else: 
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            self.dim_new = (width, height)

            # Create grid to instantiate downsized image
            zz = np.linspace(0,nz - 2,width)    
            xx = np.linspace(0,nx - 2,height)   
            X, Z = np.meshgrid(zz, xx)

            iava = np.vstack([Z.ravel(), X.ravel()])

            # create downsizing linear operator 
            Bop = pylops.signalprocessing.Bilinear(iava, (nz, nx))
            
        self.Bop = Bop
        # create downsized, blurred image
        y0 = Bop * blurred.ravel()

        # create noise
        noises = np.random.normal(0, sigma, y0.shape)

        # add noise
        y = y0 + noises
        
        # Initialize using adjoint of downsizing operator the deblur
        # xhat = Bop.H * y
        xhat = pylops.optimization.leastsquares.NormalEquationsInversion(Bop, None, y,
                                                                 epsRs=[np.sqrt(0.001)],
                                                                 returninfo=False,
                                                                 **dict(maxiter=100))

        xinit = np.real(np.fft.ifft( np.fft.fft(xhat.flatten())/np.fft.fft(blur.flatten()) )).reshape(self.dim_old) * self.H * self.W
        # xinit = (xinit - xinit.min()) / (xinit.max() - xinit.min())
        
        self.num_meas = y.size
        self.noisy = xinit.reshape(nz, nx)
        self.y = y.reshape(self.dim_new) 
        
    def batch(self, mini_batch_size):
        N = self.num_meas
        tmp = np.random.permutation(N)
        k = tmp[0:mini_batch_size]
        return k

    ## nab l(x) = B^T S^T (S B Z - y) / m
    def full_grad(self, z):
        Z_blurred = fft_blur(z, self.blur)
        Z_down = (self.Bop * Z_blurred.flatten()).reshape(self.dim_new)
        res = Z_down - self.y
        res_up = (self.Bop.H * res.flatten()).reshape(self.dim_old)
        return fft_blur(res_up, np.roll(np.flip(self.blur),1))

    # def stoch_grad(self, z, mini_batch_size):
    #     index = self.batch(mini_batch_size)
    #     res = np.zeros(self.y.shape)
    #     Z_blurred = cv2.GaussianBlur(z, (self.blur_size_x, self.blur_size_y), 0)
    #     Z_down = cv2.resize(Z_blurred, self.dim, interpolation = cv2.INTER_AREA)
    #     res.ravel()[index] = Z_down.ravel()[index] - self.y.ravel()[index]
    #     res_up = cv2.resize(res.reshape(self.dim), (self.H,self.W), interpolation = cv2.INTER_AREA)
    #     return cv2.GaussianBlur(res_up, (self.blur_size_x, self.blur_size_y), 0)
    
class PhaseRetrieval(Problem):
    def __init__(self, img_path=None, img=None, H=256, W=256, 
                       num_meas=-1, sigma=1.0,
                       lr_decay=0.999):
        super().__init__(img_path, img, H, W)
        self.sigma = sigma
        self.num_meas = num_meas
        # problem setup
        A = np.random.random((num_meas,H*W)) + np.random.random((num_meas,H*W)) * 1j

        orig = np.matrix.flatten(self.original)

        y = np.dot(A, orig)
        x_init = orig

        self.A = A
        self.noisy = x_init.reshape(H, W)
        self.y = y

    def batch(self, mini_batch_size):
        # Get batch indices in terms of (row, col)

        m = self.num_meas
        tmp = np.linspace(0, m - 1, m)
        batch_locs = np.random.choice(tmp, mini_batch_size, replace=False)

        return np.sort(batch_locs.astype(int))

    def full_grad(self, z):
        Weight = np.diag(np.divide(np.linalg.norm(np.dot(self.A,np.matrix.flatten(z)), axis=1) - np.matrix.flatten(z)),np.linalg.norm(np.dot(self.A,np.matrix.flatten(z)), axis=1))
        return (np.conj(self.A).T.dot(Weight).dot(self.A).dot(np.matrix.flatten(z))).reshape(self.H,self.W)

    def stoch_grad(self, z, mini_batch_size):
        Gamma = batch(mini_batch_size)
        A_Gamma = self.A[Gamma]
        W = np.diag(np.divide(np.linalg.norm(np.dot(A_Gamma,np.matrix.flatten(z)), axis=1) - np.matrix.flatten(z)),np.linalg.norm(np.dot(A_Gamma,np.matrix.flatten(z)), axis=1))
        return (np.conj(A_Gamma).T.dot(W).dot(A_Gamma).dot(np.matrix.flatten(z))).reshape(self.H,self.W)
        