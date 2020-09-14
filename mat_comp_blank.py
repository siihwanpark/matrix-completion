import numpy as np

class SVT():
    def __init__(self, M, mask):
        """
        M : n x m array
            true matrix
        
        mask : n x m array
            masking pattern
        """
        self.M = M
        self.mask = mask

        # TODO : initialize variables
        self.tau = 0
        self.delta = 0
        self.Y = None

    def execute(self, max_iterations):
        """
        Do the SVT algorithm
        
        Parameters
        ----------
        max_iterations : integer
            maximum iteration
        
        Returns
        ----------
        X_k : n x m array
            reconstructed matrix
        """

        # TODO : implement the SVT algorithm

        return None