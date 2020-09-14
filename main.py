import numpy as np
import matplotlib.pyplot as plt
import time

from argparse import ArgumentParser
from utils import *
from mat_comp import SVT

def main(args):
    # load the image and make mask
    image = load_image(args.image)
    mask = make_mask(image, args.mask_prob)
    save_image(mask * image, 'results/masked.png')

    # make pixel values with in range [0,1]
    normalized_image = image.astype('float64') / 255

    # matrix completion with SVT
    svt = SVT(normalized_image, mask)
    recovered_image = svt.execute(args.iter)
    
    # recover the pixel value range
    recovered_image = (recovered_image/np.max(recovered_image)) * 255
    save_image(recovered_image, 'results/recovered.png')
    
    # calculate the root mean squared error
    print("RMSE:", calc_unobserved_rmse(image, recovered_image, mask))

if __name__ == "__main__":

    argparse = ArgumentParser()
    argparse.add_argument("--image", default='data/01.png', type=str)
    argparse.add_argument("--mask_prob", default=0.5, type=float)
    argparse.add_argument("--iter", default=1000, type=int)

    args = argparse.parse_args()
    main(args)