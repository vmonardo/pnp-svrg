from PIL import Image
import numpy as np

class Problem():
    def __init__(self, img_path, H, W):
        # User specified parameters
        self.H = H                  # Height of the image
        self.W = W                  # Width of the image
        self.N = H*W                # Dimensionality of the problem

        # Load in Image to specified dimensions (H,W)
        if img_path is not None:
            tmp = np.array(Image.open(img_path).resize((H, W)))
        else:
            raise Exception('Need to pass in image path or image')

        # Normalize image such that all pixels are in rage [0,1]
        tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))
        self.X = tmp.reshape(self.N)    # Pass image as np array of specified dimensions

        # Initialize essential parameters
        # self.Y = np.empty(self.M)
        self.Xinit = np.empty_like(tmp)
    
    def get_item(self, key):
        return self.__dict__[key]

    def select_mb_indices(self, size):
        # Draw measurements uniformly at random for mini-batch stochastic gradient
        # Get batch indices in terms of (row, col)
        batch = np.zeros((1, self.H*self.W))
        batch_locs = np.random.choice(self.N, size, replace=False)
        batch[0, batch_locs] = 1
        return batch.reshape(self.H, self.W).astype(int)

    def f(self, z):
        # Method to compute the data fidelity loss at a given input
        raise NotImplementedError('Need to implement f() method')

    def grad_full(self, z):
        # Compute a full gradient w.r.t. data fidelity term
        raise NotImplementedError('Need to implement full_grad() method')

    def grad_stoch(self, z, mb_indices):
        # Compute a stochastic gradient w.r.t. data fidelity term at mini-batch indices given
        raise NotImplementedError('Need to implement stoch_grad() method')

    def grad_full_check(self):
        # Check the gradient implementation at a random value
        w = np.random.uniform(0.0, 1.0, self.N)
        delta = np.zeros(self.N)
        grad = np.zeros(self.N)
        eps = 1e-4

        for i in range(self.N):
            delta[i] = eps
            grad[i] = (self.f(w+delta) -  self.f(w - delta)) / eps   
            delta[i] = 0

        grad_comp = self.grad_full(w)
        if np.linalg.norm(grad - grad_comp) > 1e-3:
            print('Full Grad check failed!')
            print('grad diff: ', grad)
            print('grad compute: ', grad_comp)
            return False
        else:
            print('Full Grad check succeeded!')
            return True
        
    def grad_stoch_check(self):
        # check that (grad) f(w) = sum((grad) f_i(w)) at a random value
        w = np.random.uniform(0.0, 1.0, self.N)
        full_grad = self.grad_full(w)
        grad_comp = np.zeros(self.N)

        for i in range(self.N):
            mb = np.zeros(self.N)
            mb[i] = 1
            mb = mb.reshape(self.H, self.W)
            grad_comp += self.grad_stoch(w, mb).flatten()

        if np.linalg.norm(full_grad - grad_comp) > 1e-6:
            print('Stoch Grad check failed!')
            print('full grad: ', full_grad)
            print('grad comp: ', grad_comp)
            return False
        else:
            print('Stoch Grad check succeeded!')
            return True

# use this for debugging
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    height = 64
    width = 64
    noise_level = 0.01

    p = Problem(img_path='./data/Set12/01.png', H=height, W=width)
    x = p.select_mb_indices(height*width)
    print(sum(sum(x)), p.N)