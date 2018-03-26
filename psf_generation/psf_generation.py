#!/bin/env python3
# Evan Widloski - 2018-03-12
# Generate PSFs of a circular aperture at various distances and plot them

import numpy as np
from matplotlib import pyplot as plt
import itertools
from scipy.signal import convolve2d
from scipy.special import jv

np.set_printoptions(linewidth=9999, precision=2, suppress=True)


# perform fresnel approximation on s at a distance d (m) with grid x (m)
def fresnel(s, x, d, LAMBDA):
    """
    :param s: NxN image
    :param x: NxNx2 array of sample locations
    :param d: distance at which to calculate approximation
    :param LAMBDA: wavelength to perform approximation at
    :return: PSF of s at distance
    """
    h = (1 / (1j * LAMBDA * d)) * np.e**(1j * np.pi / (LAMBDA * d) * (x[:, :, 0]**2 + x[:, :, 1]**2))
    return convolve2d(h, s, mode='same')


# mask out signal outside of circle
def circ(s, x, r):
    """
    :param s: NxN image to mask out with zeroes
    :param x: NxNx2 array of sample locations
    :param r: radius of circular aperture
    :return: masked NxN image
    """
    s_temp = s.copy()
    s[np.sqrt((x*x).sum(axis=2)) > r] = 0
    return s


# generate airy disk
def airy_disk(x, distances, r):
    power = 1
    LAMBDA = 304e-10

    results = []
    for distance in distances:
        results.append(power / (np.pi * distance**2) *
                       jv(1, 2 * np.pi * r * distance /
                          (LAMBDA * np.sqrt(distance**2 +
                                            np.sqrt((x*x).sum(axis=2))
                                            )
                           )
                          )
                       )

    return results

# return list of PSFs at given distances
def generate_psfs(coords, distances, r):
    """
    :param coords: NxNx2 array of sample locations
    :param distances: list of distances
    :param r: radius of circular aperture
    :return: list of NxN complex PSFs
    """
    LAMBDA = 304e-10
    plane_wave = np.e**(1j * 2*np.pi * np.ones(coords[:, :, 0].shape))

    masked_wave = circ(plane_wave, coords, r)

    return [fresnel(masked_wave, coords, d, LAMBDA) for d in distances]

if __name__ == '__main__':

    # image side pixels
    x_pixels = 1000
    # image side length (m)
    x_length = 20e-2
    x_range = np.linspace(-x_length/2, x_length/2, x_pixels)
    coords = np.flip(np.array(list(itertools.product(x_range, x_range))), axis=1).reshape(x_pixels, x_pixels, 2)

    # distances = np.logspace(.1, 7, 8)
    distances = np.logspace(-8, -7, 6)
    # calculate PSFs of circular aperture
    # psfs = generate_psfs(coords, distances, x_length / 16)
    psfs = airy_disk(coords, distances, x_length / 2)
    # plot PSF magnitudes
    f, subplots = plt.subplots(len(psfs), 1)
    for plot, psf, d in zip(subplots, psfs, distances):
        plot.imshow(np.abs(psf), cmap='magma')
        plot.set_title('d={:.2E}'.format(d))
        plot.axis('off')

    plt.tight_layout(pad=0)
    f.subplots_adjust(wspace=.1, hspace=1)
    plt.show()
