#!/bin/env python3
# Evan Widloski - 2018-03-12
# Generate PSFs of a circular aperture at various distances and plot them

import numpy as np
from matplotlib import pyplot as plt
import itertools
from scipy.signal import convolve2d
# from psf_generation import psf_generation
from mas import psf_generation
import scipy.misc
import scipy.linalg

if __name__ == '__main__':
    # image side pixels
    x_pixels = 5
    # image side length (m)
    x_length = 20e-2
    x_range = np.linspace(-x_length/2, x_length/2, x_pixels)
    coords = np.flip(np.array(list(itertools.product(x_range, x_range))), axis=1).reshape(x_pixels, x_pixels, 2)

    # distances = np.logspace(.1, 7, 8)
    distances = np.logspace(-12, -8, 8)
    # calculate PSFs of circular aperture
    psfs = psf_generation.airy_disk(coords, distances, x_length / 2)

    H = np.empty((0, len(psfs[0].flatten())), dtype='complex')
    for psf in psfs:
        toeplitz_column = np.pad(psf.flatten(), (0, len(psf.flatten()) - 1), 'constant')
        toeplitz_row = np.zeros(len(psf.flatten()), dtype='complex')
        toeplitz_row[0] = psf.flatten()[0]
        toeplitz = scipy.linalg.toeplitz(toeplitz_column, toeplitz_row)
        H = np.append(H, toeplitz, axis=0)

    plt.imshow(np.abs(H))
    plt.show()



