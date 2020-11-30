from imports import *

def nlm(z, noise_params):
    return denoise_nl_means(z, h=noise_params['filter'], fast_mode=True, **noise_params['patch'])

def cnn(z):
    r = denoiser(torch.Tensor(z)[None][None]).squeeze().detach().cpu().numpy()
    z -= r
    return z

def tv(z, noise_params):
    return denoise_wavelet(z, multichannel=noise_params['multia'], rescale_sigma=noise_params['rescale_sigma'])

def denoiser(type, noisy, params):
    return {'nlm':nlm(noisy, params),
            'cnn':cnn(noisy),
            'bm3d':bm3d(noisy, params['noise_est']), # uses bm3d function from package
            'tv':tv(noisy, params)}[type]