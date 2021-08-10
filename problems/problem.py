from imports import *

class Problem():
    def __init__(self, img_path, img, H, W, lr_decay=1):
        if img_path is not None:
            original = np.array(Image.open(img_path).resize((H, W)))
        elif img is not None:
            original = img
        else:
            raise Exception('Need to pass in image path or image')

        original = (original - np.min(original)) / (np.max(original) - np.min(original))

        self.original = original
        self.H = H
        self.W = W
        self.lr_decay = lr_decay

    def batch(self, mini_batch_size):
        raise NotImplementedError('Need to implement batch() method')

    def full_grad(self, z):
        raise NotImplementedError('Need to implement full_grad() method')

    def stoch_grad(self, z, mini_batch_size):
        raise NotImplementedError('Need to implement stoch_grad() method')