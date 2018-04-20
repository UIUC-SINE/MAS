#!/bin/env python3
# Evan Widloski - 2018-04-18

import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def incoherent_psf(*, wavelength, diameter, smallest_zone_width,
                   plane_location, image_width):
    """
    Generate an incoherent photon-sieve PSF

    Args:
        wavelength (float): wavelength of monochromatic plane-wave source
        diameter (float): photon-sieve diameter
        smallest_zone_width (float): diameter of smallest hole in photon-sieve
        plane_location (float): distance from photon-sieve to sensor
        image_width (int): width of returned psf, in pixels (must be odd)

    Returns:
        image_width by image_width ndarray containing incoherent psf
    """

    assert image_width % 2 == 1, 'image width must be odd'

    focal_length = diameter * smallest_zone_width / wavelength
    # depth of focus
    dof = 2 * smallest_zone_width**2 / wavelength
    defocus_amount = (plane_location - focal_length) / dof
    # diffraction limited cutoff frequency (equivalent to diameter / (wavelength * focal_length))
    cutoff_freq = 1 / smallest_zone_width
    pixel_size = 1 / (2 * cutoff_freq)

    # diffraction limited bandwidth at sensor plane
    diff_limited_bandwidth = diameter / (wavelength * plane_location)
    # sampling interval in frequency domain
    delta_freq = 1 / (image_width * pixel_size)

    # defocusing parameter in Fresnel formula
    epsilon_1 = - defocus_amount * dof / (focal_length * (focal_length + defocus_amount * dof))

    # points at which to evaluate Fresnel formula
    freqs = delta_freq * np.linspace(-(image_width - 1) // 2, (image_width - 1) // 2, num=image_width)

    f_xx, f_yy = np.meshgrid(freqs, freqs)
    # circular aperture
    circ = np.sqrt(f_xx**2 + f_yy**2) <= diff_limited_bandwidth / 2
    # Fresnel approximation through aperture and free space
    h_coherent = circ * np.e**(1j * np.pi * epsilon_1 *
                             wavelength * plane_location**2 *
                             (f_xx**2 + f_yy**2))
    coherent_psf = np.fft.fftshift(np.fft.ifft2(h_coherent))
    incoherent_psf = np.abs(coherent_psf)**2
    return incoherent_psf


def load_measurements(data_file, num_copies=10):
    """
    Build measurements ndarray from HDF5 file containing psfs

    Args:
        data_file (str): path to hdf5 datafile containing incoherent PSFs.  HDF5 file should contain
                         'incoherentPsf' matrix of shape [sources]x[planes]x[psf width]x[psf height]
        num_copies (int): number of copies of each psf group to initialize array with

    Returns:
        structured ndarray with columns 'num_copies', 'num_copies_removed' and 'psfs'
    """

    import h5py

    # load psfs from file and set copies
    with h5py.File(data_file) as f:
        num_sources, num_planes, image_width, image_height = f['incoherentPsf']['value'].shape

        # measurements at each plane
        #  measurements['num_copies'] number of measurements at this distance
        #  measurements['psfs'] psfs of each source at this distance
        measurements = np.zeros(num_planes, dtype=[('num_copies', 'i'),
                                                   ('num_copies_removed', 'i'),
                                                   ('psfs', 'f', (num_sources,
                                                                  image_width,
                                                                  image_height))])
        measurements['psfs'] = np.swapaxes(f['incoherentPsf']['value'], 0, 1)

    measurements['num_copies'][:] = num_copies
    measurements['num_copies_removed'][:] = 0
    return measurements

def generate_measurements(wavelengths=np.array([33.4, 33.5, 33.6]) * 1e-9, num_planes=30,
                                               diameter=2.5e-2, smallest_zone_width=5e-6,
                                               image_width=301, num_copies=10):
    """
    Generate measurements array for CSBS algorithm

    Args:
        wavelengths (ndarray): array of source wavelengths
        num_planes (int): number of measurement planes
        diameter (float): photon-sieve diameter
        smallest_zone_width (float): smallest photon-sieve aperture diameter
        image_width (int): width of psfs (must be odd)
    """

    image_height = image_width
    num_sources = len(wavelengths)

    # focal length and depth of focus for each wavelength
    focal_lengths = diameter * smallest_zone_width / wavelengths
    dofs = 2 * smallest_zone_width**2 / wavelengths

    plane_locations = np.linspace(max(focal_lengths) - 10 * max(dofs),
                                min(focal_lengths) + 10 * min(dofs), num_planes)

    psfs = []
    # generate incoherent measurements for each wavelength and plane location
    for n, plane_location in enumerate(plane_locations):

        if n % 10 == 0:
            logging.info('{} iterations'.format(n))

        temp = []
        for wavelength in wavelengths:
            temp.append(incoherent_psf(wavelength=wavelength,
                                    diameter=diameter,
                                    smallest_zone_width=smallest_zone_width,
                                    plane_location=plane_location,
                                    image_width=image_width))
        psfs.append(temp)

    measurements = np.zeros(num_planes, dtype=[('num_copies', 'i'),
                                                ('num_copies_removed', 'i'),
                                                ('psfs', 'f', (num_sources,
                                                                image_width,
                                                                image_height))])

    measurements['psfs'] = np.array(psfs)
    measurements['num_copies'][:] = num_copies
    measurements['num_copies_removed'][:] = 0
    return measurements


if __name__ == '__main__':

    #---------- Parameter Specification ----------

    image_width = image_height = 301

    # number of measurement planes
    num_planes = 30

    # wavelengths of monochromatic sources
    wavelengths = np.array([33.4, 33.5, 33.6]) * 1e-9
    num_sources = len(wavelengths)

    # photon sieve parameters
    diameter = 2.5e-2
    smallest_zone_width = 5e-6

    #---------- PSF Generation ----------

    psf = incoherent_psf(wavelength=wavelengths[0],
                         diameter=diameter,
                         smallest_zone_width=smallest_zone_width,
                         plane_location=plane_locations[0],
                         image_width=image_width)

    import matplotlib.pyplot as plt
    plt.imshow(np.abs(psf))
    plt.show()
