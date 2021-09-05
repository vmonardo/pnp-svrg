import numpy as np
import matplotlib.pyplot as plt

def display_results(problem, output_dict, psnr_fig=None, save_results=False, save_dir='figures/'):
    # Creating directory and set base file name for saving figures
    if save_results:
        from datetime import datetime
        import os
        baseFileName = problem.prob_dir + output_dict['algo_name'] + '/'
        os.makedirs(baseFileName, exist_ok=True)

    # Display output image
    output_img = output_dict['z'].reshape(problem.H, problem.W)
    out_fig = plt.figure(figsize=(3,3))
    plt.imshow(output_img, cmap=problem.color_map, vmin=0, vmax=1)
    plt.title('Output Image')
    plt.xticks([])
    plt.yticks([])
    if save_results:
        fileName = baseFileName + 'output.eps'
        out_fig.savefig(fileName, transparent = True, bbox_inches = 'tight', pad_inches = 0)
    plt.show()

    # Plot time vs PSNR
    if psnr_fig is None:
        psnr_fig = plt.figure(figsize=(8,8))
        psnr_ax = psnr_fig.add_subplot(1,1,1)

    tArray = output_dict['time_per_iter']
    psnrArray = output_dict['psnr_per_iter']
    psnr_ax.plot(np.cumsum(tArray), psnrArray, "b", linewidth=3, label=str(output_dict['algo_name']))
    psnr_ax.plot(np.cumsum(tArray)[::30], psnrArray[::30], "b*", markersize=10)
    psnr_ax.set(xlabel='time (s)', ylabel='PSNR (dB)')
    psnr_ax.legend()
    psnr_ax.grid()
    plt.show()
    
    psnr_fig.tight_layout()
    if save_results:
        fileName = baseFileName + 'psnr_over_time.eps'
        psnr_fig.savefig(fileName, transparent = True, bbox_inches = 'tight', pad_inches = 0)

    # Output metrics table
    print('Output PSNR: {0:3.1f}\tChange in PSNR: {2:3.2f}\tGradient Time: {3:3.2f}\tDenoising Time: {3:3.2f}'.format(
        psnrArray[-1], psnrArray[-1] - psnrArray[0], output_dict['gradient_time'], output_dict['denoise_time']
    ))
    # print('| {0:3.1f} |\t| {1:3.1f} |\t| {2:3.2f} |\t| {3:3.2f} |'.format(psnrArray[-1], psnrArray[-1] - psnrArray[0], output_dict['gradient_time'], output_dict['denoise_time']))
    return psnr_fig