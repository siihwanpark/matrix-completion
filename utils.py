import numpy as np

from PIL import Image
from scipy.stats import bernoulli

def load_image(path):
    """
    load the image from the given path.
    Return value is a numpy 2D array.
    """
    image = np.asarray(Image.open(path))

    return image

def save_image(image, path):
    """
    save the given image to the given path.
    """
    img = Image.fromarray(image.astype('uint8'))
    img.save(path)

def make_mask(image, prob_masked=0.5):
    """
    Generate a randomly generated binary mask for the given image.
    Note that 1 denotes observed, and 0 denotes unobserved.
    """
    return 1 - bernoulli.rvs(p=prob_masked, size=image.shape)

def calc_unobserved_rmse(A, A_hat, mask):
    """
    Calculate RMSE on all unobserved entries in mask, for true matrix UVᵀ.

    Parameters
    ----------
    A : m x n array
        true matrix
        
    A_hat : m x n array
        estimated matrix

    mask : m x n array
        matrix with entries zero (if missing) or one (if present)

    Returns:
    --------
    rmse : float
        root mean squared error over all unobserved entries
    """
    pred = np.multiply(A_hat, mask)
    truth = np.multiply(A, mask)
    cnt = np.sum(mask)
    return (np.linalg.norm(pred - truth, "fro") ** 2 / cnt) ** 0.5