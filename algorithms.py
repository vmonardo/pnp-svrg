from imports import *

def pnp_svrg(problem, denoiser, eta, tt, T2, mini_batch_size, verbose=True):
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []
    
    # Main PnP-SVRG routine
    z = np.copy(problem.noisy)
    zs = [z]

    denoiser.t = 0

    i = 0

    elapsed = time.time()

    # outer loop
    while (time.time() - elapsed) < tt:
        start_time = time.time()

        # Full gradient at reference point
        mu = problem.full_grad(z)

        # Initialize reference point
        w = np.copy(z) 

        time_per_iter.append(time.time() - start_time)
        psnr_per_iter.append(peak_signal_noise_ratio(problem.original, z))

        # inner loop
        for j in range(T2): 
            if (time.time() - elapsed) >= tt:
                return z, time_per_iter, psnr_per_iter, zs

            # start timing
            start_time = time.time()

            # calculate stochastic variance-reduced gradient (SVRG)
            v = problem.stoch_grad(z, mini_batch_size) - problem.stoch_grad(w, mini_batch_size) + mu

            # Gradient update
            z -= (eta*denoiser.lr_decay**denoiser.t)*v

            if verbose:
                print("After gradient update: " + str(i) + " " + str(j) + " " + str(peak_signal_noise_ratio(problem.original, z)))

            # Denoise
            z = denoiser.denoise(noisy=z)
            
            zs.append(z)

            denoiser.t += 1

            # stop timing
            time_per_iter.append(time.time() - start_time)
            psnr_per_iter.append(peak_signal_noise_ratio(problem.original, z))

            if verbose:
                print("After denoising update: " + str(i) + " " + str(j) + " " + str(psnr_per_iter[-1]))
                print()
        
        i += 1

    # output denoised image, time stats, psnr stats
    return z, time_per_iter, psnr_per_iter, zs


def pnp_gd(problem, denoiser, eta, tt, verbose=True):
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []

    # Main PnP GD routine
    z = np.copy(problem.noisy)
    zs = [z]

    denoiser.t = 0

    i = 0

    elapsed = time.time()

    while (time.time() - elapsed) < tt:
        # start timing
        start_time = time.time()

        # calculate full gradient
        v = problem.full_grad(z)

        # Gradient update
        z -= (eta*denoiser.lr_decay**denoiser.t)*v

        # Denoise
        z = denoiser.denoise(noisy=z)
        
        zs.append(z)

        denoiser.t += 1

        # Log timing
        time_per_iter.append(time.time() - start_time)

        psnr_per_iter.append(peak_signal_noise_ratio(problem.original, z))

        if verbose:
            print(str(i) + " " + str(psnr_per_iter[-1]))

        i += 1

    return z, time_per_iter, psnr_per_iter, zs


def pnp_sgd(problem, denoiser, eta, tt, mini_batch_size, verbose=True):
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []

    # Main PnP SGD routine
    z = np.copy(problem.noisy)
    zs = [z]

    denoiser.t = 0

    i = 0

    elapsed = time.time()

    while (time.time() - elapsed) < tt:
        # start timing
        start_time = time.time()

        # calculate stochastic gradient
        v = problem.stoch_grad(z, mini_batch_size)

        # Gradient update
        z -= (eta*denoiser.lr_decay**denoiser.t)*v

        # Denoise
        z = denoiser.denoise(noisy=z)

        zs.append(z)

        denoiser.t += 1

        # Log timing
        time_per_iter.append(time.time() - start_time)

        psnr_per_iter.append(peak_signal_noise_ratio(problem.original, z))

        if verbose:
            print(str(i) + " " + str(psnr_per_iter[-1]))

        i += 1

    return z, time_per_iter, psnr_per_iter, zs


def pnp_lsvrg(problem, denoiser, eta, tt, mini_batch_size, prob_update=0.1, verbose=True):
    # Initialize logging variables
    time_per_iter = []
    psnr_per_iter = []

    # Main PnP LSVRG routine
    z = np.copy(problem.noisy)
    zs = [z]

    denoiser.t = 0

    i = 0

    w = np.copy(z)

    elapsed = time.time()
    
    # calculate full gradient
    start_time = time.time()
    mu = problem.full_grad(z)
    time_per_iter.append(time.time() - start_time)
    psnr_per_iter.append(peak_signal_noise_ratio(problem.original, z))

    while (time.time() - elapsed) < tt:
        # start timing
        start_time = time.time()

        # calculate stochastic gradient
        v = problem.stoch_grad(z, mini_batch_size) - problem.stoch_grad(w, mini_batch_size) + mu

        # Gradient update
        z -= (eta*denoiser.lr_decay**denoiser.t)*v

        # Denoise
        z = denoiser.denoise(noisy=z)

        zs.append(z)

        denoiser.t += 1

        # update reference point with probability prob_update
        if np.random.random() < prob_update:
            w = np.copy(z)
            mu = problem.full_grad(z)

        # Log timing
        time_per_iter.append(time.time() - start_time)

        psnr_per_iter.append(peak_signal_noise_ratio(problem.original, z))

        if verbose:
            print(str(i) + " " + str(psnr_per_iter[-1]))

        i += 1

    return z, time_per_iter, psnr_per_iter, zs