from problems.problem import Problem
from imports import *

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
        self.blurred = blurred

        if scale_percent == 100:
            width, height = nz, nx
            self.dim_new = self.dim_old
            Bop = pylops.Identity(width*height)
        else: 
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            self.dim_new = (width, height)

            # Create grid to instantiate downsized image
            zz = np.linspace(0.00001,nz - 1.00001,width)    
            xx = np.linspace(0.00001,nx - 1.00001,height)   
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
        # blur_hat= Bop * blur.ravel()
        # deblurred_hat = np.real(np.fft.ifft( np.fft.fft(y.flatten())/np.fft.fft(blur_hat.flatten()) )).reshape(self.dim_new) * self.dim_new[0] * self.dim_new[1]
        # xinit = Bop.H * deblurred_hat.ravel()

        D2op = pylops.Laplacian((nz, nx), weights=(1, 1), dtype='float64')

        xhat = pylops.optimization.leastsquares.NormalEquationsInversion(Bop, [D2op], y.ravel(),
                                                                 epsRs=[np.sqrt(0.01)],
                                                                 returninfo=False,
                                                                 **dict(maxiter=100))

        self.xhat = xhat.reshape(self.dim_old)
        xinit = np.real(np.fft.ifft( np.fft.fft(xhat)/np.fft.fft(blur.flatten()) )).reshape(self.dim_old) * self.H * self.W
        # xinit = (xinit - xinit.min()) / (xinit.max() - xinit.min())
        
        # xhat = Bop.H * y
        # xinit = np.real(np.fft.ifft( np.fft.fft(xhat)/np.fft.fft(blur.flatten()) )).reshape(self.dim_old) * self.H * self.W

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

    ## nab l(x) = B^T S^T (S B Z - y) / m
    def stoch_grad(self, z, mini_batch_size):
        index = self.batch(mini_batch_size)
        res = np.zeros(self.y.shape)
        Z_blurred = fft_blur(z, self.blur)
        # Z_blurred = cv2.GaussianBlur(z, (self.blur_size_x, self.blur_size_y), 0)
        Z_down = (self.Bop * Z_blurred.flatten()).reshape(self.dim_new)
        # Z_down = cv2.resize(Z_blurred, self.dim, interpolation = cv2.INTER_AREA)
        res.ravel()[index] = Z_down.ravel()[index] - self.y.ravel()[index]
        res_up = (self.Bop.H * res.flatten()).reshape(self.dim_old)
        # res_up = cv2.resize(res.reshape(self.dim), (self.H,self.W), interpolation = cv2.INTER_AREA)
        return fft_blur(res_up, np.roll(np.flip(self.blur),1))
    