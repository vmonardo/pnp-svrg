from imports import *

def denoise(kind, noisy, problem):
    if kind == 'cnn':
        return (noisy - (problem.params['cnn_decay']**problem.params['t'])*problem.params['cnn'](torch.Tensor(noisy)[None][None].to(problem.params['device'])).squeeze().detach().cpu().numpy())
    elif kind == 'nlm':
        return denoise_nl_means(noisy, h=problem.params['filter']*problem.params['filter_decay']**problem.params['t'], fast_mode=True, **problem.params['patch'])
    elif kind == 'tv':
        return denoise_wavelet(noisy, multichannel=problem.params['multia'], rescale_sigma=problem.params['rescale_sigma'])
    elif kind == 'bm3d':
        return bm3d(noisy, problem.params['noise_est']*problem.params['filter_decay']**problem.params['t'])