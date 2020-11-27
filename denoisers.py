from imports import *

def svrg(img_path, denoiser, eta, T1, T2, batch_size):
    d = process_img(img_path) 

    time_per_iter = []
    psnr_per_iter = []
    
    '''
    PnP SVRG routine
    '''

    z = d['noisy']

    zs = [z]

    # outer loop
    for i in range(T1):
        # Gradient at reference point
        mu = full_grad(z, d['mask'], d['y'])# / np.count_nonzero(d['mask']) 

        w = np.copy(z) # Initialize reference point

        start_iter = time.time()

        # inner loop
        for j in range(T2):

            # Get batch index(indices) in terms of (row, col)
            ind = get_batch(batch_size, d['mask']) 

            # calculate stochastic variance-reduced gradient
            v = stoch_grad(z, ind, d['y']) / batch_size - stoch_grad(w, ind, d['y']) / batch_size + mu
            
            # take gradient step
            z = z - eta*v

            '''
            NORMALIZE
            '''
            # z_min = np.min(z)
            # z_max = np.max(z)
            #z = (z - z_min)/(z_max - z_min)

            if isinstance(denoiser, str) and denoiser == 'nlm':
                z = denoise_nl_means(z, h=0.015, fast_mode=True, **d['patch'])
            elif isinstance(denoiser, Denoiser):
                '''
                SCALE
                '''
                #scale_range = 1.0 + denoiser.sigma/255.0/2.0
                #scale_shift = (1 - scale_range)/2.0
                #z = z * scale_range + scale_shift
                
                out = denoiser(torch.Tensor(z)[None][None]).squeeze().detach().cpu().numpy()
                z -= out

                '''
                UN-SCALE + UN-NORMALIZE
                '''
                #z = (z - scale_shift)/scale_range
                #z = z * (z_max - z_min) + z_min
                
            zs.append(z)

            print("After denoising: " + str(i) + " " + str(j) + " " + str(peak_signal_noise_ratio(d['original'], z)))

        # calculate time difference and PSNR
        time_per_iter.append(time.time() - start_iter)
        psnr_per_iter.append(peak_signal_noise_ratio(d['original'], z))
    
    # output denoised image, time stats, psnr stats
    return z, time_per_iter, psnr_per_iter, zs

        
        
       




