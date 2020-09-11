import numpy as np

class SVT():
    def __init__(self, M, mask):
        self.M = M
        self.mask = mask
        self.tau = 5 * np.sum(self.M.shape) / 2
        self.delta = 1.2 * np.prod(self.M.shape) / np.sum(self.mask)
        self.Y = np.zeros_like(self.M)

    def execute(self, max_iterations):
        for k in range(max_iterations):
            U, S, V = np.linalg.svd(self.Y, full_matrices=False)

            S = np.maximum(S - self.tau, 0)

            X = np.linalg.multi_dot([U, np.diag(S), V])
            self.Y += self.delta * self.mask * (self.M - X)

            recon_error = np.linalg.norm(self.mask * (X - self.M)) / np.linalg.norm(self.mask * self.M)
            print("[%d/%d] reconstruction_error : %.4f"%(k+1, max_iterations, recon_error))

        return X