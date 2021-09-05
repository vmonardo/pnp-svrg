import numpy as np
import matplotlib.pyplot as plt

BASECOLORS = ["b", "g", "r", "c", "m", "k"] # excluding white and yellow

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
        plot_count = 0                          # Initialize number of lines on the plot

    plot_count = psnr_ax.lines // 2
    if plot_count > 6:
        raise Exception('Add more colors to BASECOLORS. Current plot count: '.format(plot_count))
        
    line_color = BASECOLORS[plot_count]
    tArray = output_dict['time_per_iter']
    psnrArray = output_dict['psnr_per_iter']
    psnr_ax.plot(np.cumsum(tArray), psnrArray, line_color, linewidth=3, label=str(output_dict['algo_name']))
    psnr_ax.plot(np.cumsum(tArray)[::30], psnrArray[::30], line_color + "*", markersize=10)
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
    return psnr_fig, plot_count