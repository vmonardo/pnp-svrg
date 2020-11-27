from imports import *

def show_multiple(images, ax=plt):
    cols = len(images)
    fig, axes = ax.subplots(ncols=cols, figsize=(7,3))
    
    for i, image in enumerate(images):
        image = image.cpu().detach().numpy()
        image = (image - image.min())/(image.max() - image.min())
        h = axes[i].imshow(image.squeeze(), cmap='gray')
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
    return np.real(np.fft.ifft2(res))


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
    return np.real(np.fft.ifft2(res))


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