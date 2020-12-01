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
 
def myimshow(image, ax=plt):
    image = image.cpu().detach().numpy()
    image = (image - image.min())/(image.max() - image.min())
    h = ax.imshow(image.squeeze(), cmap='gray')
    return h

def psnr_display(img_path, output, title):
    ORIG = np.array(Image.open(img_path).resize((256,256))) / 255.0
    original = (ORIG - np.min(ORIG)) / (np.max(ORIG) - np.min(ORIG))

    psnr_out = peak_signal_noise_ratio(original, output)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    svrg_plot = plt.imshow(output, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f"%s, PSNR = {psnr_out:0.2f}" % title)
    ax.axis('off')


def full_grad(z, MASK, meas):
    # Input: 
    # z, optimization iterate
    # MASK, observed fourier measurements
    # meas, measurements = F(X) + w
    # Output:
    # Full gradient at z

    # real grad
    # H, W = z.shape[:2]
    res = np.fft.fft2(z) * MASK
    index = np.nonzero(MASK)
    res[index] = res[index] - meas[index]
    return np.real(np.fft.ifft2(res)) / np.count_nonzero(MASK)


def stoch_grad(z, IND, meas):
    # Input:
    # z, optimization iterate
    # meas, measurements = F(X) + w
    # batch_index, indices to update
    # Output:
    # stochastic gradient at z for measurements in B
    # H, W = z.shape[:2]
    # batch gradient update
    res = IND * (np.fft.fft2(z) - meas)
    return np.real(np.fft.ifft2(res)) / np.count_nonzero(IND)


def get_batch(B, MASK):
    H, W = MASK.shape[:2]
    batch = np.zeros((1,H*W))
    tmp = np.linspace(0, H*W - 1, H*W)
    one_locs = tmp[np.matrix.flatten(MASK) == 1].astype(int)
    batch_locs = np.random.choice(one_locs, B, replace=False)
    batch[0, batch_locs] = 1

    # find nonzero batch indices
    return batch.reshape(H,W).astype(int)


def gif(images):
    fig = plt.figure()

    im = plt.imshow(images[0], cmap='gray', vmin=0, vmax=1)

    def init():
        im.set_data(images[0])
        return im
        
    def animate(i):
        im.set_data(images[i])
        return im

    anim = FuncAnimation(fig, animate, init_func=init, frames=range(len(images)), interval=40)

    return HTML(anim.to_html5_video())

def create_problem(img_path='./data/Set12/13.jpg', H=256, W=256, sample_prob=0.5, sigma=1.0,  # general params 
                   filter_size=0.015, patch_size=5, patch_distance=6, multichannel=True,      # NLM params
                   network_type='DnCNN', device='cpu',                                        # CNN params
                   multia=True, rescale_sigma=True,                                           # TV params
                   noise_est=0.015):                                                          # BM3D params

    original = np.array(Image.open(img_path).resize((H, W)))
    original = (original - np.min(original)) / (np.max(original) - np.min(original))

    np.random.seed(0)
    mask = np.random.choice([0, 1], size=(H, W), p=[1 - sample_prob, sample_prob])
    indices = np.transpose(np.nonzero(mask))

    forig = np.fft.fft2(original)
    np.random.seed(0)
    noises = np.random.normal(0, sigma, (H, W))

    y0 = forig + noises
    y = np.multiply(mask, y0)

    x_init = np.absolute(np.fft.ifft2(y))
    # x_init = (x_init - np.min(x_init)) / (np.max(x_init) - np.min(x_init))

    cnn = Denoiser(net=eval(network_type)(17), 
                   experiment_name='exp1_flickr30k_' + network_type, 
                   data=False, sigma=30, batch_size=10).net.to(device)

    return {'noisy': x_init,
            'mask': mask,
            'y': y,
            'original': original,
            'H': H,
            'W': W,
            'indices': indices,
            'filter' : filter_size,
            'patch' : dict(patch_size=patch_size, patch_distance=patch_distance, multichannel=multichannel),
            'cnn' : cnn,
            'multia' : multia,
            'rescale_sigma' : rescale_sigma,
            'noise_est' : noise_est}
