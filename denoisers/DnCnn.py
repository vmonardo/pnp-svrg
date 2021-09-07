#!/usr/bin/env python
# coding=utf-8
try:
    from .denoiser import Denoise
    from .cnn.cnn import Denoiser, DnCnn
except:
    from denoiser import Denoise
    from cnn.cnn import Denoiser, DnCNN
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CNNDenoiser(Denoise):
    def __init__(self, decay=1,
                       sigma_est=40, device='cuda'):
        super().__init__()

        # Set user defined parameters
        self.decay = decay
        self.sigma_est = sigma_est
        self.device = device

        self.cnn = Denoiser(net=DnCNN(17), experiment_name='exp_' + str(sigma_est), 
                            data=False, sigma=sigma_est, batch_size=20).net.to(device)

    def denoise(self, noisy):
        self.t += 1
        return (noisy - (self.decay**self.t)*self.cnn(torch.Tensor(noisy)[None][None].to(self.device)).squeeze().detach().cpu().numpy())

### See cnn.py for more info

if __name__=='__main__':
    import torch
    from utils import show_multiple
    import matplotlib.pyplot as plt
    from cnn.cnn import Denoiser, FlickrSet, DnCNN
    from cnn.spectral import spectral_norm

    # print('cuda' if torch.cuda.is_available() else 'cpu')

    sigma = 40
    test_set = FlickrSet(mode='test', sigma=sigma, image_size=(40,40))

    # sample clean test image
    test_img = test_set[1]

    # print("Norm difference:", torch.norm(test_img[0] - test_img[1]).item())
    print("PSNR:", 10*torch.log10(len(test_img[0].reshape(-1)) / torch.norm(test_img[0]-test_img[1])**2).item(), '\n\n')
    print("Noisy             |        Clean")
    show_multiple([test_img[0], test_img[1]])

    fig, ax = plt.subplots()
    im = ax.imshow(test_img[2], cmap='gray')
    fig.colorbar(im, orientation='horizontal')
    # plt.show()

    network_type = 'DnCNN'
    denoiser = Denoiser(net=DnCNN(17), 
                        experiment_name='exp_' + str(sigma), 
                        data=True,
                        sigma=sigma,
                        batch_size=20)

    #                     #train network 

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(7,6))

    denoiser.run(num_epochs=20, fig=fig, axes=axes, noisy_img=test_img[0])

    # evaluate performance on test set

    denoiser.evaluate()
