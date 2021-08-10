from problems.problem import Problem
from imports import *

class PhaseRetrieval(Problem):
    def __init__(self, img_path=None, img=None, H=256, W=256, 
                       num_meas=-1, sigma=1.0,
                       lr_decay=0.999):
        super().__init__(img_path, img, H, W)
        self.sigma = sigma
        self.num_meas = num_meas
        # problem setup
        A = np.random.random((num_meas,H*W)) + np.random.random((num_meas,H*W)) * 1j

        orig = np.matrix.flatten(self.original)

        y = np.dot(A, orig)
        x_init = orig

        self.A = A
        self.noisy = x_init.reshape(H, W)
        self.y = y

    def batch(self, mini_batch_size):
        # Get batch indices in terms of (row, col)

        m = self.num_meas
        tmp = np.linspace(0, m - 1, m)
        batch_locs = np.random.choice(tmp, mini_batch_size, replace=False)

        return np.sort(batch_locs.astype(int))

    def full_grad(self, z):
        Weight = np.diag(np.divide(np.linalg.norm(np.dot(self.A,np.matrix.flatten(z)), axis=1) - np.matrix.flatten(z)),np.linalg.norm(np.dot(self.A,np.matrix.flatten(z)), axis=1))
        return (np.conj(self.A).T.dot(Weight).dot(self.A).dot(np.matrix.flatten(z))).reshape(self.H,self.W)

    def stoch_grad(self, z, mini_batch_size):
        Gamma = batch(mini_batch_size)
        A_Gamma = self.A[Gamma]
        W = np.diag(np.divide(np.linalg.norm(np.dot(A_Gamma,np.matrix.flatten(z)), axis=1) - np.matrix.flatten(z)),np.linalg.norm(np.dot(A_Gamma,np.matrix.flatten(z)), axis=1))
        return (np.conj(A_Gamma).T.dot(W).dot(A_Gamma).dot(np.matrix.flatten(z))).reshape(self.H,self.W)