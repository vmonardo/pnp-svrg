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
        h = axis.imshow(image.squeeze(), cmap='gray', vmin=0, vmax=1)
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

def grad(z, indices, meas, F, N):
    index = np.nonzero(indices)
    
    res = np.zeros((N,N), dtype=complex)
    
    F_i = F[index[0],:]
    F_j = F[index[1],:]
    
    res[index] = ((F_i @ z * F_j).sum(-1) - meas[index])
    
    return (np.real(np.conj(F) @ res @ np.conj(F.T))/N**2)/len(index[0])

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

    anim = FuncAnimation(fig, animate, init_func=init, frames=range(len(images)), interval=60)

    return HTML(anim.to_html5_video())

def create_problem(img_path=None, img=None, H=256, W=256, sample_prob=0.5, sigma=1.0,         # general params
                   filter_size=0.015, patch_size=5, patch_distance=6, multichannel=True,      # NLM params
                   cnn_sigma=40, device='cuda',                                               # CNN params
                   multia=True, rescale_sigma=True,                                           # TV params
                   noise_est=0.015,                                                           # BM3D params
                   lr_decay=0.999, filter_decay=0.999, cnn_decay=0.9):                        # decay params                   

    if img_path is not None:
        original = np.array(Image.open(img_path).resize((H, W)))
    elif img is not None:
        original = img
    else:
        raise Exception('Need to pass in image path or image')

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

    cnn = Denoiser(net=DnCNN(17), 
                   experiment_name='exp_' + str(cnn_sigma), 
                   data=False, sigma=cnn_sigma, batch_size=20).net.to(device)

    i, j = np.meshgrid(np.arange(H), np.arange(H))
    omega = np.exp(-2*math.pi*1J/H)
    F = np.power(omega, i*j)

    return {'noisy': x_init,
            'mask': mask,
            'y': y,
            'original': original,
            'H': H,
            'W': W,
            'F' : F,
            'indices': indices,
            'filter' : filter_size,
            'patch' : dict(patch_size=patch_size, patch_distance=patch_distance, multichannel=multichannel),
            'cnn' : cnn,
            'device' : device,
            'multia' : multia,
            'rescale_sigma' : rescale_sigma,
            'noise_est' : noise_est,
            't' : 0,
            'lr_decay' : lr_decay,
            'filter_decay' : filter_decay,
            'cnn_decay' : cnn_decay}

def denoise_rgb(img_path):
    from algorithms import pnp_svrg

    def denoise_slice(params):
        return pnp_svrg(params, denoiser='cnn', eta=20e3, tt=20, T2=10, 
                        mini_batch_size=int(20e3), verbose=False)[0]
    
    img = np.array(Image.open(img_path).resize((256,256)), dtype=float)

    slice0 = create_problem(img=img[:,:,0])
    slice1 = create_problem(img=img[:,:,1])
    slice2 = create_problem(img=img[:,:,2])

    noisy = np.rollaxis(np.array([slice0['noisy'], slice1['noisy'], slice2['noisy']]), 0, 3)
    original = np.rollaxis(np.array([slice0['original'], slice1['original'], slice2['original']]), 0, 3)

    out0, out1, out2 = denoise_slice(slice0), denoise_slice(slice1), denoise_slice(slice2)
    
    denoised = np.rollaxis(np.array([out0, out1, out2]), 0, 3)
    
    return original, noisy, denoised
