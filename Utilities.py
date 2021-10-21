import numpy as np
import matplotlib.pyplot as plt
import itertools as it

def display_results(problem, output_dict, save_results=False, save_dir='figures/', show_figs=False):
    # Creating directory and set base file name for saving figures
    if save_results:
        from datetime import datetime
        import os
        if problem.prob_dir:
            baseFileName = problem.prob_dir + output_dict['algo_name'] + '/'
        else:
            baseFileName = save_dir
        os.makedirs(baseFileName, exist_ok=True)

    # Display output image
    output_img = output_dict['z'].reshape(problem.H, problem.W)
    out_fig = plt.figure(figsize=(6,6))
    plt.imshow(output_img, cmap=problem.color_map, vmin=0, vmax=1)
    plt.title('Output Image')
    plt.xticks([])
    plt.yticks([])
    if save_results:
        fileName = baseFileName + 'output.eps'
        out_fig.savefig(fileName, transparent = True, bbox_inches = 'tight', pad_inches = 0)
    if show_figs:
        plt.show()

    # Plot time vs PSNR
    psnr_fig = plt.figure(figsize=(6,6))
    psnr_ax = psnr_fig.add_subplot(1,1,1)
        
    tArray = output_dict['time_per_iter']
    psnrArray = output_dict['psnr_per_iter']
    psnr_ax.plot(np.cumsum(tArray), psnrArray, "b", linewidth=3, label=str(output_dict['algo_name']))
    psnr_ax.plot(np.cumsum(tArray)[::30], psnrArray[::30], "b*", markersize=10)
    psnr_ax.set(xlabel='time (s)', ylabel='PSNR (dB)')
    psnr_ax.legend()
    psnr_ax.grid()
    psnr_fig.tight_layout()
    
    if show_figs:
        plt.show()
    
    
    if save_results:
        fileName = baseFileName + 'psnr_over_time.eps'
        psnr_fig.savefig(fileName, transparent = True, bbox_inches = 'tight', pad_inches = 0)

    # Output metrics table
    print('Output PSNR: {0:3.1f}\tChange in PSNR: {2:3.2f}\tGradient Time: {3:3.2f}\tDenoising Time: {3:3.2f}'.format(
        psnrArray[-1], psnrArray[-1] - psnrArray[0], output_dict['gradient_time'], output_dict['denoise_time']
    ))
    if save_results:
        import csv 
        csv_fn = baseFileName + 'output.csv'
        with open(csv_fn, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['Output PSNR', 'Change in PSNR', 'Gradient Time', 'Denoising Time'])
            writer.writerow([   np.around(psnrArray[-1], decimals=1),
                                np.around(psnrArray[-1] - psnrArray[0], decimals=2),
                                np.around(output_dict['gradient_time'], decimals=2),
                                np.around(output_dict['denoise_time'], decimals=2) ])
    return psnr_ax

if __name__ == '__main__':
    import sys
    sys.path.append('problems/')
    sys.path.append('denoisers/')
    from problems.CSMRI import CSMRI
    from algorithms import *

    im_height, im_width = 256, 256  # Image dimensions
    samp_rate = 0.5                 # Pick a number 0 < SR <= 1
    sigma_true = 5.0                # Select std dev of AWGN

    main_problem = CSMRI('./data/Set12/13.png', H=im_height, W=im_width, sample_prob=samp_rate, sigma=sigma_true)
    main_problem.display(show_measurements=False, save_results=False)

    from denoisers.NLM import NLMDenoiser
    denoiser = NLMDenoiser(sigma_est=1, patch_size=4, patch_distance=5)

    results_svrg = pnp_svrg(main_problem, denoiser=denoiser, eta=5e-3, tt=1, T2=50, mini_batch_size=100, verbose=True)
    master_psnr_ax = display_results(main_problem, results_svrg, save_results=False)
    print(master_psnr_ax)

    results_sgd = pnp_sgd(main_problem, denoiser=denoiser, eta=5e-3, tt=1, mini_batch_size=100, verbose=True)
    master_psnr_ax = display_results(main_problem, results_sgd, save_results=False)

    plt.show()


        
    