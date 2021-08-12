# Denoise class to handle image denoising within an algorithm
class Denoise():
    def __init__(self):
        self.t = 0

    def denoise(self, noisy):
        raise NotImplementedError('Need to implement denoise() method')