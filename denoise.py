from imports import *

class Denoise():
    def __init__(self):
        self.t = 0

    def denoise(self, noisy):
        raise NotImplementedError('Need to implement denoise() method')

class CNNDenoiser(Denoise):
    def __init__(self, cnn_decay=0.9,
                       cnn_sigma=40, device='cuda'):
        super().__init__()

        # denoiser setup
        self.cnn_decay = cnn_decay
        self.cnn_sigma = cnn_sigma
        self.device = device

        self.cnn = Denoiser(net=DnCNN(17), experiment_name='exp_' + str(cnn_sigma), 
                            data=False, sigma=cnn_sigma, batch_size=20).net.to(device)

    def denoise(self, noisy):
        return (noisy - (self.cnn_decay**self.t)*self.cnn(torch.Tensor(noisy)[None][None].to(self.device)).squeeze().detach().cpu().numpy())

class NLMDenoiser(Denoise):
    def __init__(self, filter_decay=0.999,
                       filter_size=0.015, patch_size=5, patch_distance=6, multichannel=True):
        super().__init__()

        # denoiser setup
        self.filter_decay = filter_decay
        self.filter_size = filter_size
        self.patch = dict(patch_size=patch_size, patch_distance=patch_distance, multichannel=multichannel)

    def denoise(self, noisy):
        return denoise_nl_means(noisy, h=self.filter_size*self.filter_decay**self.t, fast_mode=True, **self.patch)

class TVDenoiser(Denoise):
    def __init__(self, multia=True, rescale_sigma=True):
        super().__init__()

        # denoiser setup
        self.multia = multia
        self.rescale_sigma = rescale_sigma

    def denoise(self, noisy):
        return denoise_wavelet(noisy, multichannel=self.multia, rescale_sigma=self.rescale_sigma)

class BM3DDenoiser(Denoise):
    def __init__(self, filter_decay=0.999,
                       noise_est=0.015):
        super().__init__()

        # denoiser setup
        self.filter_decay = filter_decay
        self.noise_est = noise_est

    def denoise(self, noisy):
        return bm3d(noisy, self.noise_est*self.filter_decay**self.t)