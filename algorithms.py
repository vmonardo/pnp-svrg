from imports import *

def pnp_svrg(params, denoiser, eta, T1, T2, mini_batch_size, verbose=True):
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []
    
    # Main PnP-SVRG routine
    z = np.copy(params['noisy'])
    zs = [z]

    params['t'] = 0

    # outer loop
    for i in range(T1):
        # Full gradient at reference point
        mu = grad(z, params['mask'], params['y'], params['F'], params['H'])

        w = np.copy(z) # Initialize reference point

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

            # Denoise
            z = denoise(kind=denoiser, noisy=z, params=params)
            
            zs.append(z)

            params['t'] += 1

            # stop timing
            time_per_iter.append(time.time() - start_time)

            psnr_per_iter.append(peak_signal_noise_ratio(params['original'], z))

            if verbose:
                print(str(i) + " " + str(j) + " " + str(psnr_per_iter[-1]))

    # output denoised image, time stats, psnr stats
    return z, time_per_iter, psnr_per_iter, zs


def pnp_gd(params, denoiser, eta, T, verbose=True):
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []

    # Main PnP GD routine
    z = np.copy(params['noisy'])
    zs = [z]

    params['t'] = 0

    for i in range(T):
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

    return z, time_per_iter, psnr_per_iter, zs


def pnp_sgd(params, denoiser, eta, T, mini_batch_size, verbose=True):
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []

    # Main PnP SGD routine
    z = np.copy(params['noisy'])
    zs = [z]

    params['t'] = 0

    for i in range(T):
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

    return z, time_per_iter, psnr_per_iter, zs


def pnp_lsvrg(params, denoiser, eta, T, mini_batch_size, prob_update=0.1, verbose=True):
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []

    # Main PnP LSVRG routine
    z = np.copy(params['noisy'])
    zs = [z]

    params['t'] = 0

    w = np.copy(z)

    for i in range(T):
        # start timing
        start_time = time.time()

        # calculate full gradient
        mu = grad(z, params['mask'], params['y'], params['F'], params['H'])

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

        # Log timing
        time_per_iter.append(time.time() - start_time)

        psnr_per_iter.append(peak_signal_noise_ratio(params['original'], z))

        if verbose:
            print(str(i) + " " + str(psnr_per_iter[-1]))

    return z, time_per_iter, psnr_per_iter, zs
       




