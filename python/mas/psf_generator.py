#!/bin/env python3
# Evan Widloski - 2018-04-18

import numpy as np
import logging
from decimal import Decimal
import math

logging.basicConfig(level=logging.INFO, format='%(message)s')

class Measurements():
    """A class for holding PSFs and state data during csbs iterations

    Args:
        psfs (ndarray): an array holding the 2D psfs.  shape: (num_planes, num_sources)
        num_copies (int): number of repeated measurements to initialize with
        measurement_wavelengths (ndarray): measurement wavelength at each. shape: (num_planes)

    Attributes:
        psfs (ndarray): an array holding the 2D psfs
        psf_ffts (ndarray): an array holding the 2D psf ffts
        num_copies (int): number of repeated measurements to initialize with
        copies (ndarray): array containing current number of measurements at each location
        measurement_wavelengths (ndarray): an array holding the wavelength at each measurement plane
        copies_history (list): list containing indices of removed measurement plane indices for each iteration
    """
    def __init__(self, *, psfs, num_copies, measurement_wavelengths):

        assert psfs.shape[0] == len(measurement_wavelengths), "`psfs` and `wavelengths` shapes do not match"

        self.psfs = psfs
        self.psf_ffts = np.fft.fft2(psfs)
        self.num_copies = num_copies
        self.copies = np.ones((len(measurement_wavelengths))) * num_copies
        self.measurement_wavelengths = measurement_wavelengths
        self.image_width = psfs.shape[2]
        self.copies_history = []


def incoherent_psf(*, source_wavelength, measurement_wavelength,
                   diameter, smallest_zone_width, image_width):
    """
    Generate an incoherent photon-sieve PSF

    Args:
        wavelength (float): wavelength of monochromatic plane-wave source
        diameter (float): photon-sieve diameter
        smallest_zone_width (float): diameter of smallest hole in photon-sieve
        image_width (int): width of returned psf, in pixels (must be odd)

    Returns:
        image_width by image_width ndarray containing incoherent psf
    """

    assert image_width % 2 == 1, 'image width must be odd'

    focal_length = diameter * smallest_zone_width / source_wavelength
    plane_location = diameter * smallest_zone_width / measurement_wavelength
    # depth of focus
    dof = 2 * smallest_zone_width**2 / source_wavelength
    defocus_amount = (plane_location - focal_length) / dof
    # diffraction limited cutoff frequency
    # (equivalent to diameter / (source_wavelength * focal_length))
    cutoff_freq = 1 / smallest_zone_width
    pixel_size = 1 / (2 * cutoff_freq)

    # diffraction limited bandwidth at sensor plane
    diff_limited_bandwidth = diameter / (source_wavelength * plane_location)
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
                             source_wavelength * plane_location**2 *
                             (f_xx**2 + f_yy**2))
    coherent_psf = np.fft.fftshift(np.fft.ifft2(h_coherent))
    incoherent_psf = np.abs(coherent_psf)**2
    return incoherent_psf


# def load_measurements(data_file, num_copies=10):
#     """
#     Build measurements ndarray from HDF5 file containing psfs

#     Args:
#         data_file (str): path to hdf5 datafile containing incoherent PSFs.  HDF5 file should contain
#                          'incoherentPsf' matrix of shape [sources]x[planes]x[psf width]x[psf height]
#         num_copies (int): number of copies of each psf group to initialize array with

#     Returns:
#         structured ndarray with columns 'num_copies', 'num_copies_removed' and 'psfs'
#     """

#     import h5py

#     # load psfs from file and set copies
#     with h5py.File(data_file) as f:
#         num_sources, num_planes, image_width, image_height = f['incoherentPsf']['value'].shape

#         measurements = np.zeros(num_planes, dtype=[('num_copies', 'i'),
#                                                    ('num_copies_removed', 'i'),
#                                                    ('psfs', 'f', (num_sources,
#                                                                   image_width,
#                                                                   image_height))])
#         psfs = np.swapaxes(f['incoherentPsf']['value'], 0, 1)


#     measurements = Measurements(psfs=psfs, num_copies=num_copies)
#     return measurements

def generate_measurements(
        source_wavelengths=np.array([33.4, 33.5, 33.6]) * 1e-9,
        measurement_wavelengths=30, diameter=2.5e-2, smallest_zone_width=5e-6,
        image_width=301, num_copies=10
):
    """
    Generate measurements array for CSBS algorithm

    Args:
        source_wavelengths (ndarray): array of source wavelengths
        measurement_wavelengths (int/ndarray): number of measurement wavelengths, or an array of wavelengths to measure at
        diameter (float): photon-sieve diameter
        smallest_zone_width (float): smallest photon-sieve aperture diameter
        image_width (int): width of psfs (must be odd)

    Returns:
        Measurements object with psfs and csbs data
    """

    image_height = image_width
    num_sources = len(source_wavelengths)
    source_wavelengths = np.array(list(map(Decimal, map(str, source_wavelengths))))
    smallest_zone_width = Decimal(smallest_zone_width)
    diameter = Decimal(diameter)

    if type(measurement_wavelengths) is int:
        # find planes/spacing closest to user input so a sample occurs exactly at each
        # focal length
        num_wavelengths = measurement_wavelengths

        def decimal_gcd(l, prec=5):

            i = 0
            j = 1
            while len(set(l)) != 1:
                print(l)
                l[i] = l[i] % l[j]
                l[i] = l[j] if l[i] == 0 else l[i]
                i += 1
                j += 1
                i %= len(l)
                j %= len(l)

            return l[0]

        spacing = decimal_gcd(np.diff(sorted(source_wavelengths)))

        # focal length and depth of focus for each wavelength
        focal_lengths = diameter * smallest_zone_width / source_wavelengths
        dofs = 2 * smallest_zone_width**2 / source_wavelengths
        print("focal_lengths:", focal_lengths)
        print("spacing:", spacing)

        # calculate start and end wavelengths using DOF
        approx_start = diameter * smallest_zone_width / (max(focal_lengths) + 10 * max(dofs))
        approx_end = diameter * smallest_zone_width / (min(focal_lengths) - 10 * min(dofs))
        # align wavelengths to grid
        aligned_start = min(source_wavelengths) - math.ceil(
            (min(source_wavelengths) - approx_start) / spacing
        ) * spacing
        aligned_end = math.ceil(
            (approx_end - max(source_wavelengths)) / spacing
        ) * spacing + max(source_wavelengths)

        # calculate the number of planes needed to span aligned enpoints
        minimum_planes = (aligned_end - aligned_start) / spacing
        print('minimum_planes', minimum_planes)

        # round number of measurement locations up to the multiple
        # closest to user input
        num_wavelengths = math.ceil(num_wavelengths / minimum_planes) * minimum_planes
        print('num_wavelengths', num_wavelengths)

        measurement_wavelengths = np.linspace(
            float(aligned_start),
            float(aligned_end),
            int(num_wavelengths) + 1
        )

    psfs = np.empty((0, num_sources, image_width, image_width))

    # generate incoherent measurements for each wavelength and plane location
    for n, measurement_wavelength in enumerate(measurement_wavelengths):

        if n % 10 == 0:
            logging.info('{} iterations'.format(n))

        psf_group = np.empty((0, image_width, image_width))
        for source_wavelength in source_wavelengths:
            psf = incoherent_psf(
                source_wavelength=float(source_wavelength),
                diameter=float(diameter),
                smallest_zone_width=float(smallest_zone_width),
                measurement_wavelength=measurement_wavelength,
                image_width=image_width
            )

            psf_group = np.append(psf_group, [psf], axis=0)
        psfs = np.append(psfs, [psf_group], axis=0)

    measurements = Measurements(
        psfs=psfs,
        num_copies=num_copies,
        measurement_wavelengths=measurement_wavelengths
    )

    return measurements


if __name__ == '__main__':

    #---------- Parameter Specification ----------

    image_width = image_height = 301

    # number of measurement planes
    num_planes = 30

    # wavelengths of monochromatic sources
    source_wavelengths = np.array([33.4, 33.5, 33.6]) * 1e-9
    num_sources = len(source_wavelengths)

    # photon sieve parameters
    diameter = 2.5e-2
    smallest_zone_width = 5e-6

    #---------- PSF Generation ----------

    psf = incoherent_psf(
        source_wavelength=source_wavelengths[0],
        measurement_wavelength=source_wavelengths[1],
        diameter=diameter,
        smallest_zone_width=smallest_zone_width,
        image_width=image_width
    )

    import matplotlib.pyplot as plt
    plt.imshow(np.abs(psf))
    plt.show()
