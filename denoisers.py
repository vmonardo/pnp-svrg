from imports import *

def denoise(kind, noisy, params):
    if kind == 'cnn':
        return (noisy - params['cnn'](torch.Tensor(noisy)[None][None]).squeeze().detach().cpu().numpy())
    elif kind == 'nlm':
        return denoise_nl_means(noisy, h=params['filter'], fast_mode=True, **params['patch'])
    elif kind == 'tv':
        return denoise_wavelet(noisy, multichannel=params['multia'], rescale_sigma=params['rescale_sigma'])
    elif kind == 'bm3d':
        return bm3d(noisy, params['noise_est'])