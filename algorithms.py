from imports import *

def pnp_svrg(params, denoiser, eta, T1, T2, mini_batch_size, verbose=True):
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []
    
    # Main PnP-SVRG routine
    z = params['noisy']
    zs = [z]

    # outer loop
    for i in range(T1):
        # Gradient at reference point
        mu = full_grad(z, params['mask'], params['y'])

        w = np.copy(z) # Initialize reference point

        # inner loop
        for j in range(T2):  
            # start timing
            start_time = time.time()

            # Get batch index(indices) in terms of (row, col)
            ind = get_batch(mini_batch_size, params['mask'])

            # calculate stochastic variance-reduced gradient
            v = stoch_grad(z, ind, params['y']) - stoch_grad(w, ind, params['y']) + mu
            
            # Gradient update
            z -= eta*v

            # Denoise
            z = denoise(kind=denoiser, noisy=z, params=params)
            
            zs.append(z)

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
    z = params['noisy']
    zs = [z]

    for i in range(T):
        # start timing
        start_time = time.time()

        # calculate full gradient
        v = full_grad(z, params['mask'], params['y'])

        # Gradient update
        z -= eta*v

        # Denoise
        z = denoise(kind=denoiser, noisy=z, params=params)
        
        zs.append(z)

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
    z = params['noisy']
    zs = [z]

    for i in range(T):
        # start timing
        start_time = time.time()

        # Get minibatch
        ind = get_batch(mini_batch_size, params['mask'])

        # calculate stochastic gradient
        v = stoch_grad(z, ind, params['y'])

        # Gradient update
        z -= eta*v

        # Denoise
        z = denoise(kind=denoiser, noisy=z, params=params)

        zs.append(z)

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
    z = params['noisy']
    zs = [z]

    w = np.copy(z)

    for i in range(T):
        # start timing
        start_time = time.time()

        # calculate full gradient
        mu = full_grad(z, params['mask'], params['y'])

        # Get minibatch
        ind = get_batch(mini_batch_size, params['mask'])

        # calculate stochastic gradient
        v = stoch_grad(z, ind, params['y']) - stoch_grad(w, ind, params['y']) + mu

        # Gradient update
        z -= eta*v

        # Denoise
        z = denoise(kind=denoiser, noisy=z, params=params)

        zs.append(z)

        # update reference point with probability prob_update
        if np.random.random() < prob_update:
            w = np.copy(z)

        # Log timing
        time_per_iter.append(time.time() - start_time)

        psnr_per_iter.append(peak_signal_noise_ratio(params['original'], z))

        if verbose:
            print(str(i) + " " + str(psnr_per_iter[-1]))

    return z, time_per_iter, psnr_per_iter, zs
       




