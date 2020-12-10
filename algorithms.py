from imports import *

def pnp_svrg(params, denoiser, eta, tt, T2, mini_batch_size, verbose=True):
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []
    
    # Main PnP-SVRG routine
    z = np.copy(params['noisy'])
    zs = [z]

    params['t'] = 0

    elapsed = time.time()
    i = 0

    # outer loop
    while (time.time() - elapsed) < tt:
        # Full gradient at reference point
        start_time = time.time()

        mu = grad(z, params['mask'], params['y'], params['F'], params['H'])

        w = np.copy(z) # Initialize reference point

        time_per_iter.append(time.time() - start_time)
        psnr_per_iter.append(peak_signal_noise_ratio(params['original'], z))

        # inner loop
        for j in range(T2):  
            # start timing
            start_time = time.time()

            # Get batch index(indices) in terms of (row, col)
            ind = get_batch(mini_batch_size, params['mask'])

            # calculate stochastic variance-reduced gradient (SVRG)
            v = grad(z, ind, params['y'], params['F'], params['H']) - grad(w, ind, params['y'], params['F'], params['H']) + mu

            # Gradient update
            z -= (eta*params['lr_decay']**params['t'])*v

            if verbose:
                print("After gradient update: " + str(i) + " " + str(j) + " " + str(peak_signal_noise_ratio(params['original'], z)))

            # Denoise
            z = denoise(kind=denoiser, noisy=z, params=params)
            
            zs.append(z)

            params['t'] += 1

            # stop timing
            time_per_iter.append(time.time() - start_time)
            psnr_per_iter.append(peak_signal_noise_ratio(params['original'], z))

            if verbose:
                print("After denoising update: " + str(i) + " " + str(j) + " " + str(psnr_per_iter[-1]))
        
        i += 1

    # output denoised image, time stats, psnr stats
    return z, time_per_iter, psnr_per_iter, zs


def pnp_gd(params, denoiser, eta, tt, verbose=True):
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []

    # Main PnP GD routine
    z = np.copy(params['noisy'])
    zs = [z]

    params['t'] = 0

    elapsed = time.time()
    i = 0

    while (time.time() - elapsed) < tt:
        # start timing
        start_time = time.time()

        # calculate full gradient
        v = grad(z, params['mask'], params['y'], params['F'], params['H'])

        # Gradient update
        z -= (eta*params['lr_decay']**params['t'])*v

        # Denoise
        z = denoise(kind=denoiser, noisy=z, params=params)
        
        zs.append(z)

        params['t'] += 1

        # Log timing
        time_per_iter.append(time.time() - start_time)

        psnr_per_iter.append(peak_signal_noise_ratio(params['original'], z))

        if verbose:
            print(str(i) + " " + str(psnr_per_iter[-1]))

        i += 1

    return z, time_per_iter, psnr_per_iter, zs


def pnp_sgd(params, denoiser, eta, tt, mini_batch_size, verbose=True):
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []

    # Main PnP SGD routine
    z = np.copy(params['noisy'])
    zs = [z]

    params['t'] = 0

    elapsed = time.time()
    i = 0

    while (time.time() - elapsed) < tt:
        # start timing
        start_time = time.time()

        # Get minibatch
        ind = get_batch(mini_batch_size, params['mask'])

        # calculate stochastic gradient
        v = grad(z, ind, params['y'], params['F'], params['H'])

        # Gradient update
        z -= (eta*params['lr_decay']**params['t'])*v

        # Denoise
        z = denoise(kind=denoiser, noisy=z, params=params)

        zs.append(z)

        params['t'] += 1

        # Log timing
        time_per_iter.append(time.time() - start_time)

        psnr_per_iter.append(peak_signal_noise_ratio(params['original'], z))

        if verbose:
            print(str(i) + " " + str(psnr_per_iter[-1]))

        i += 1

    return z, time_per_iter, psnr_per_iter, zs


def pnp_lsvrg(params, denoiser, eta, tt, mini_batch_size, prob_update=0.1, verbose=True):
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []

    # Main PnP LSVRG routine
    z = np.copy(params['noisy'])
    zs = [z]

    params['t'] = 0

    elapsed = time.time()
    i = 0

    w = np.copy(z)

    start_time = time.time()
    # calculate full gradient
    mu = grad(z, params['mask'], params['y'], params['F'], params['H'])
    time_per_iter.append(time.time() - start_time)
    psnr_per_iter.append(peak_signal_noise_ratio(params['original'], z))

    while (time.time() - elapsed) < tt:
        # start timing
        start_time = time.time()

        # Get minibatch
        ind = get_batch(mini_batch_size, params['mask'])

        # calculate stochastic gradient
        v = grad(z, ind, params['y'], params['F'], params['H']) - grad(w, ind, params['y'], params['F'], params['H']) + mu

        # Gradient update
        z -= (eta*params['lr_decay']**params['t'])*v

        # Denoise
        z = denoise(kind=denoiser, noisy=z, params=params)

        zs.append(z)

        params['t'] += 1

        # update reference point with probability prob_update
        if np.random.random() < prob_update:
            w = np.copy(z)
            mu = grad(z, params['mask'], params['y'], params['F'], params['H'])

        # Log timing
        time_per_iter.append(time.time() - start_time)

        psnr_per_iter.append(peak_signal_noise_ratio(params['original'], z))

        if verbose:
            print(str(i) + " " + str(psnr_per_iter[-1]))

        i += 1

    return z, time_per_iter, psnr_per_iter, zs
       




