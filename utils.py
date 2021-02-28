from imports import *

def show_multiple(images, ax=plt):
    cols = len(images)
    fig, axes = ax.subplots(ncols=cols, figsize=(7,3))
    
    for i, image in enumerate(images):
        image = image.cpu().detach().numpy()
        image = (image - image.min())/(image.max() - image.min())
        h = axes[i].imshow(image.squeeze(), cmap='gray')
    return h

def show_grid(images, titles, rows=3, cols=2, figsize=(7,3), ax=plt):
    fig, axes = ax.subplots(nrows=rows, ncols=cols, figsize=figsize)

    assert(len(axes.flatten()) == len(images))

    for axis, image, title in zip(axes.flatten(), images, titles):
        image = (image - image.min())/(image.max() - image.min())
        axis.set_title(title)
        h = axis.imshow(image.squeeze(), cmap='gray')

    fig.tight_layout()

    return h

def psnr_display(output, title, img_path=None, img=None):
    if img_path is not None:
        ORIG = np.array(Image.open(img_path).resize((256,256)))
        original = (ORIG - np.min(ORIG)) / (np.max(ORIG) - np.min(ORIG))
    elif img is not None:
        original = img
    else:
        raise Exception('Need to pass in image path or image')

    psnr_out = peak_signal_noise_ratio(original, output)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    svrg_plot = plt.imshow(output, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f"%s, PSNR = {psnr_out:0.2f}" % title)
    ax.axis('off')


def gif(images):
    fig = plt.figure()

    im = plt.imshow(images[0], cmap='gray', vmin=0, vmax=1)

    def init():
        im.set_data(images[0])
        return im
        
    def animate(i):
        im.set_data(images[i])
        return im

    anim = FuncAnimation(fig, animate, init_func=init, frames=range(len(images)), interval=60)

    return HTML(anim.to_html5_video())

class Problem():
    def __init__(self, img_path=None, img=None, H=256, W=256,                                     # general params
                       filter_size=0.015, patch_size=5, patch_distance=6, multichannel=True,      # NLM params
                       cnn_sigma=40, device='cuda',                                               # CNN params
                       multia=True, rescale_sigma=True,                                           # TV params
                       noise_est=0.015,                                                           # BM3D params
                       lr_decay=0.999, filter_decay=0.999, cnn_decay=0.9):                        # decay params                       
        self.params = {'prob_type' : None}

        if img_path is not None:
            original = np.array(Image.open(img_path).resize((H, W)))
        elif img is not None:
            original = img
        else:
            raise Exception('Need to pass in image path or image')

        original = (original - np.min(original)) / (np.max(original) - np.min(original))

        cnn = Denoiser(net=DnCNN(17), 
                       experiment_name='exp_' + str(cnn_sigma), 
                       data=False, sigma=cnn_sigma, batch_size=20).net.to(device)
        
        self.params['original'] = original
        self.params['H'] = H
        self.params['W'] = W
        self.params['filter_size'] = filter_size
        self.params['patch'] = dict(patch_size=patch_size, patch_distance=patch_distance, multichannel=multichannel)
        self.params['cnn'] = cnn
        self.params['device'] = device
        self.params['multia'] = multia
        self.params['rescale_sigma'] = rescale_sigma
        self.params['noise_est'] = noise_est
        self.params['t'] = 0
        self.params['lr_decay'] = lr_decay
        self.params['filter_decay'] = filter_decay
        self.params['cnn_decay'] = cnn_decay
    
    def setup_reconstruct(self, sample_prob=0.5, sigma=1.0):
        if self.params['prob_type'] != None:
            raise Exception('Problem type already defined: %s' % self.params['prob_type'])
        
        self.params['prob_type'] = 'reconstruct'

        np.random.seed(0)
        mask = np.random.choice([0, 1], size=(self.params['H'], self.params['W']), p=[1-sample_prob, sample_prob])

        forig = np.fft.fft2(self.params['original'])
        np.random.seed(0)
        noises = np.random.normal(0, sigma, (self.params['H'], self.params['W']))

        y0 = forig + noises
        y = np.multiply(mask, y0)
        x_init = np.absolute(np.fft.ifft2(y))

        i, j = np.meshgrid(np.arange(self.params['H']), np.arange(self.params['W']))
        omega = np.exp(-2*math.pi*1J/self.params['H'])
        F = np.power(omega, i*j)

        self.params['noisy'] = x_init
        self.params['mask'] = mask
        self.params['y'] = y
        self.params['F'] = F


    def setup_deblur(self, kernel_path=None, kernel=None):
        if self.params['prob_type'] != None:
            raise Exception('Problem type already defined: %s' % self.params['prob_type'])

        self.params['prob_type'] = 'deblur'

        if kernel_path is not None:
            blur = np.array(Image.open(kernel_path).resize((self.params['H'], self.params['W'])))
        elif kernel is not None:
            blur = kernel
        else:
            raise Exception('Need to pass in blur kernel path or image')


    def grad(self, grad_type):
        if self.params['prob_type']  == 'reconstruct':
            
            if grad_type == 'stoch':
                return self.stoch_reconstruct
            
            elif grad_type == 'full':
                return self.full_reconstruct

        elif self.params['prob_type'] == 'deblur':
            
            if grad_type == 'stoch':
                return self.stoch_deblur
            
            elif grad_type == 'full':
                return self.full_deblur

    def batch_reconstruct(self, mini_batch_size):
        mask = self.params['mask']

        # Get batch index(indices) in terms of (row, col)

        H, W = mask.shape[:2]
        batch = np.zeros((1, H*W))
        tmp = np.linspace(0, H*W - 1, H*W)
        one_locs = tmp[np.matrix.flatten(mask) == 1].astype(int)
        batch_locs = np.random.choice(one_locs, mini_batch_size, replace=False)
        batch[0, batch_locs] = 1

        return batch.reshape(H, W).astype(int)

    def full_reconstruct(self, z):
        F = self.params['F']
        N = self.params['H']
        meas = self.params['y']

        index = np.nonzero(self.params['mask'])

        res = np.zeros((N, N), dtype=complex)
        
        F_i = F[index[0],:]
        F_j = F[index[1],:]
        
        res[index] = ((F_i @ z * F_j).sum(-1) - meas[index])
        
        return (np.real(np.conj(F) @ res @ np.conj(F.T))/N**2)/len(index[0])

    def stoch_reconstruct(self, z, mini_batch_size):
        F = self.params['F']
        N = self.params['H']
        meas = self.params['y']

        index = np.nonzero(self.batch_reconstruct(mini_batch_size))
        
        res = np.zeros((N, N), dtype=complex)
        
        F_i = F[index[0],:]
        F_j = F[index[1],:]
        
        res[index] = ((F_i @ z * F_j).sum(-1) - meas[index])
        
        return (np.real(np.conj(F) @ res @ np.conj(F.T))/N**2)/len(index[0])

    def batch_deblur(self, mini_batch_size):
        pass

    def full_deblur(self, z):
        pass

    def stoch_deblur(self, z, mini_batch_size):
        pass

   

def denoise_rgb(img_path):
    from algorithms import pnp_svrg

    def denoise_slice(problem):
        return pnp_svrg(problem, denoiser='cnn', eta=20e3, tt=20, T2=10, 
                        mini_batch_size=int(20e3), verbose=False)[0]
    
    img = np.array(Image.open(img_path).resize((256,256)), dtype=float)

    slice0 = Problem(img=img[:,:,0])
    slice1 = Problem(img=img[:,:,1])
    slice2 = Problem(img=img[:,:,2])

    slice0.setup_reconstruct()
    slice1.setup_reconstruct()
    slice2.setup_reconstruct()

    noisy = np.rollaxis(np.array([slice0.params['noisy'], slice1.params['noisy'], slice2.params['noisy']]), 0, 3)
    original = np.rollaxis(np.array([slice0.params['original'], slice1.params['original'], slice2.params['original']]), 0, 3)

    out0, out1, out2 = denoise_slice(slice0), denoise_slice(slice1), denoise_slice(slice2)
    
    denoised = np.rollaxis(np.array([out0, out1, out2]), 0, 3)
    
    return original, noisy, denoised
