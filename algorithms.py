from imports import *

def pnp_svrg(d, denoiser, denoise_params, eta, T1, T2, mini_batch_size, verbose=True):
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []
    
    # Main PnP-SVRG routine
    z = d['noisy']
    zs = [z]

    # outer loop
    for i in range(T1):
        # Gradient at reference point
        mu = full_grad(z, d['mask'], d['y'])

        w = np.copy(z) # Initialize reference point

        start_iter = time.time()

        # inner loop
        for j in range(T2):

            # Get batch index(indices) in terms of (row, col)
            ind = get_batch(mini_batch_size, d['mask'])

            # start timing
            start_inner = time.time()

            # calculate stochastic variance-reduced gradient
            v = stoch_grad(z, ind, d['y']) - stoch_grad(w, ind, d['y']) + mu
            
            # take gradient step
            z = z - eta*v

            # Denoise
            if isinstance(denoiser, str) and denoiser == 'nlm':
                z = denoise_nl_means(z, h=denoise_params['filter'], **denoise_params['patch'])
            elif isinstance(denoiser, Denoiser):
                out = denoiser(torch.Tensor(z)[None][None]).squeeze().detach().cpu().numpy()
                z -= out
            zs.append(z)

            # stop timing
            stop_inner = time.time()
            time_per_iter.append(stop_inner - start_inner)
            psnr_per_iter.append(peak_signal_noise_ratio(d['original'], z))

            if verbose:
                print(str(i) + " " + str(j) + " " + str(peak_signal_noise_ratio(d['original'], z)))

    # output denoised image, time stats, psnr stats
    return z, time_per_iter, psnr_per_iter, zs


def pnp_gd(d, denoiser, denoise_params, eta, T, verbose=True):
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []
    t1 = 0

    # Main PnP GD routine
    z = d['noisy']
    zs = [z]

    for i in range(T):
        start_iter = time.time()

        # Gradient Update
        v = full_grad(z, d['mask'], d['y'])
        z = z - eta * v

        # print("After gradient: " + str(i) + " " + str(j) + " " + str(peak_signal_noise_ratio(d['original'], z)))

        # Denoising
        if isinstance(denoiser, str) and denoiser == 'nlm':
            z = denoise_nl_means(z, h=denoise_params['filter'], **denoise_params['patch'])
        elif isinstance(denoiser, Denoiser):
            out = denoiser(torch.Tensor(z)[None][None]).squeeze().detach().cpu().numpy()
            z -= out
        zs.append(z)

        # Log timing
        stop_iter = time.time()
        time_per_iter.append(stop_iter - start_iter)

        psnr_per_iter.append(peak_signal_noise_ratio(d['original'], z))

        if verbose:
            # Display PSNR at each iteration
            print(str(i) + " " + str(j) + " " + str(peak_signal_noise_ratio(d['original'], z)))
        t1 += 1
    return z, time_per_iter, psnr_per_iter, zs


def pnp_sgd(d, denoiser, denoise_params, eta, T, mini_batch_size, verbose=True):
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []
    t2 = 0

    # Main PnP SGD routine
    z = d['noisy']
    zs = [z]

    for i in range(T):
        # Update variables
        ind = get_batch(mini_batch_size, d['mask'])

        # start timing
        start_iter = time.time()
        v = stoch_grad(z, ind, d['y'])
        z = z - eta * v

        # Denoising
        if isinstance(denoiser, str) and denoiser == 'nlm':
            z = denoise_nl_means(z, h=denoise_params['filter'], **denoise_params['patch'])
        elif isinstance(denoiser, Denoiser):
            out = denoiser(torch.Tensor(z)[None][None]).squeeze().detach().cpu().numpy()
            z -= out
        zs.append(z)

        # end timing
        stop_iter = time.time()
        time_per_iter.append(stop_iter - start_iter)
        psnr_per_iter.append(peak_signal_noise_ratio(d['original'], z))

        if verbose:
            print(str(i) + " " + str(j) + " " + str(peak_signal_noise_ratio(d['original'], z)))
        t2 += 1
    return z, time_per_iter, psnr_per_iter, zs


def pnp_lsvrg(d, denoiser, denoise_params, eta, T, mini_batch_size, prob_update=0.1, verbose=True):
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []
    t4 = 0

    # Main PnP SVRG routine
    z = d['noisy']
    zs = [z]

    w = np.copy(z)
    for i in range(T):
        # outer loop
        mu = full_grad(z, d['mask'], d['y'])  # Average gradient

        # inner loop
        ind = get_batch(mini_batch_size, d['mask'])   # Get batch index(indices) in terms of (row, col)

        # start timing
        start_iter = time.time()

        v = stoch_grad(z, ind, d['y']) / mini_batch_size - stoch_grad(w, ind, d['y']) / mini_batch_size + mu
        z = z - eta * v

        # Denoising
        if isinstance(denoiser, str) and denoiser == 'nlm':
            z = denoise_nl_means(z, h=denoise_params['filter'], **denoise_params['patch'])
        elif isinstance(denoiser, Denoiser):
            out = denoiser(torch.Tensor(z)[None][None]).squeeze().detach().cpu().numpy()
            z -= out
        zs.append(z)

        # update reference point with probability prob_update
        if np.random.random() < prob_update:
            w = np.copy(z)

        # end timing
        stop_iter = time.time()
        time_per_iter.append(stop_iter - start_iter)

        if verbose:
            print(str(i) + " " + str(j) + " " + str(peak_signal_noise_ratio(d['original'], z)))

        psnr_per_iter.append(peak_signal_noise_ratio(d['original'], z))
        t4 += 1
    return z, time_per_iter, psnr_per_iter, zs
       




