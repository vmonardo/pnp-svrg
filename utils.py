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

def psnr_display(output, title, img_path=None, img=None, H=256, W=256):
    if img_path is not None:
        ORIG = np.array(Image.open(img_path).resize((H,W)))
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


def denoise_rgb(img_path):
    from algorithms import pnp_svrg
    from denoise import CNNDenoiser
    from problem import CSMRI

    def denoise_slice(problem, denoiser):
        return pnp_svrg(problem, denoiser=denoiser, eta=20e3, tt=20, T2=10, 
                        mini_batch_size=int(20e3), verbose=False)[0]
    
    img = np.array(Image.open(img_path).resize((256,256)), dtype=float)

    slice0 = CSMRI(img=img[:,:,0])
    slice1 = CSMRI(img=img[:,:,1])
    slice2 = CSMRI(img=img[:,:,2])

    noisy = np.rollaxis(np.array([slice0.noisy, slice1.noisy, slice2.noisy]), 0, 3)
    original = np.rollaxis(np.array([slice0.original, slice1.original, slice2.original]), 0, 3)

    denoiser = CNNDenoiser()

    out0, out1, out2 = denoise_slice(slice0, denoiser), denoise_slice(slice1, denoiser), denoise_slice(slice2, denoiser)
    
    denoised = np.rollaxis(np.array([out0, out1, out2]), 0, 3)
    
    return original, noisy, denoised
