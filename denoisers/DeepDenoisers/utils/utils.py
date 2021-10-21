# Utils used in this project.
# Authors: Jialin Liu (UCLA math, danny19921123@gmail.com)

import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

# ---- load the model based on the type and sigma (noise level) ---- 
def load_model(model_type, sigma):
    path = "./denoisers/DeepDenoisers/Pretrained_models/" + model_type + "_noise" + str(sigma) + ".pth"
    if model_type == "DnCNN":
        from denoisers.DeepDenoisers.model.models import DnCNN
        net = DnCNN(channels=1, num_of_layers=17)
        model = nn.DataParallel(net).cuda()
    elif model_type == "SimpleCNN":
        from denoisers.DeepDenoisers.model.SimpleCNN_models import DnCNN
        model = DnCNN(1, num_of_layers = 4, lip = 0.0, no_bn = True).cuda()
    elif model_type == "RealSN_DnCNN":
        from denoisers.DeepDenoisers.model.realSN_models import DnCNN
        net = DnCNN(channels=1, num_of_layers=17)
        model = nn.DataParallel(net).cuda()
    elif model_type == "RealSN_SimpleCNN":
        from denoisers.DeepDenoisers.model.SimpleCNN_models import DnCNN
        model = DnCNN(1, num_of_layers = 4, lip = 1.0, no_bn = True).cuda()
    else:
        from denoisers.DeepDenoisers.model.realSN_models import DnCNN
        net = DnCNN(channels=1, num_of_layers=17)
        model = nn.DataParallel(net).cuda()

    model.load_state_dict(torch.load(path))
    model.eval()
    return model

# ---- calculating PSNR (dB) of x ----- 
def psnr(x,im_orig):
    xout = (x - np.min(x)) / (np.max(x) - np.min(x))
    im_orig_out = (im_orig - np.min(im_orig)) / (np.max(im_orig) - np.min(im_orig))
    # norm1 = np.sum((np.absolute(im_orig)) ** 2)
    # norm2 = np.sum((np.absolute(x - im_orig)) ** 2)
    # psnr = 10 * np.log10( norm1 / norm2 )
    return peak_signal_noise_ratio(xout, im_orig_out)
