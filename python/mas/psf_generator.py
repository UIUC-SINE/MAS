#!/bin/env python3
# Evan Widloski - 2018-04-18

import numpy as np
import logging
from decimal import Decimal
import math
import functools

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


def photon_sieve(*, sieve_diameter, smallest_hole_diameter,
                 hole_diameter_to_zone_width, open_area_ratio):
    """
    Generate photon sieve hole locations and diameters

    Args:
        sieve_diameter (float): photon sieve diameter
        smallest_hole_diameter (float): diameter of holes on outermost zone
        hole_diameter_to_zone_width (float): ratio of hole diameter to zone
        open_area_ratio (float): ratio of hole area to total sieve area 

    Returns:
        A list of dictionaries representing each 'open' zone with the following keys:

        hole_diameter (float): diameter of holes in this zone
        hole_coordinates (list): list of tuples containing x and y hole coordinates
        inner_radius (float): inner radius of zone accounting for hole overlap
        inner_radius (float): outer radius of zone accounting for hole overlap
    """

    num_white_zones = np.floor(sieve_diameter**2 / (8 * sieve_diameter * smallest_hole_diameter))
    zone_radii = np.sqrt(2 * sieve_diameter * smallest_hole_diameter * np.arange(1, num_white_zones + 1))
    zone_widths = sieve_diameter * smallest_hole_diameter / (2 * zone_radii)
    hole_diameters = hole_diameter_to_zone_width * zone_widths
    hole_counts = np.round(8 * open_area_ratio * zone_widths * zone_radii / (hole_diameters**2))

    white_zones = []
    # generate each white zone
    for hole_diameter, hole_count, zone_radius in zip(hole_diameters, hole_counts, zone_radii):
        white_zone = {}
        white_zone['hole_diameter'] = hole_diameter
        white_zone['hole_coordinates'] = []
        white_zone['inner_radius'] = zone_radius - hole_diameter / 2
        white_zone['outer_radius'] = zone_radius + hole_diameter / 2
        # generate each hole
        for theta in 2 * np.pi * np.arange(hole_count) / hole_count:
            white_zone['hole_coordinates'].append(
                (
                    zone_radius * np.cos(theta),
                    zone_radius * np.sin(theta)
                )
            )

        white_zones.append(white_zone)

    return white_zones


def incoherent_psf(
        *,
        white_zones,
        source_wavelength,
        measurement_wavelength,
        sieve_diameter,
        smallest_hole_diameter,
        image_width,
        source_distance=float('Inf')
):
    """
    Generate an incoherent photon-sieve PSF

    Args:
        white_zones (list): list of dictionaries repesenting each white_zone
        source_wavelength (float): wavelength of monochromatic plane-wave source
        sieve_diameter (float): photon-sieve diameter
        smallest_hole_diameter (float): diameter of smallest hole in photon-sieve
        image_width (int): width of returned psf, in pixels (must be odd)
        source_distance (float): distance to source

    Returns:
        image_width by image_width ndarray containing incoherent psf
    """

    print(source_wavelength, measurement_wavelength, sieve_diameter,
          smallest_hole_diameter, image_width)

    assert image_width % 2 == 1, 'image width must be odd'

    def a_func(x, y):
        radius = np.sqrt(x**2 + y**2)
        # return True if point falls inside hole
        for white_zone in white_zones:
            if radius >= white_zone['inner_radius']:
                if radius < white_zone['outer_radius']:
                    theta = np.arctan2(y, x)
                    closest_hole = int(
                        np.round(len(white_zone['hole_coordinates']) * theta / (2 * np.pi))
                    )
                    # check if point falls in hole
                    if np.sqrt(
                            (x - white_zone['hole_coordinates'][closest_hole][0])**2 +
                            (y - white_zone['hole_coordinates'][closest_hole][1])**2
                    ) < white_zone['hole_diameter'] / 2:
                        return True
                    else:
                        return False
            else:
                return False
        return False

    a = np.vectorize(a_func)

    focal_length = sieve_diameter * smallest_hole_diameter / source_wavelength
    plane_distance = sieve_diameter * smallest_hole_diameter / measurement_wavelength
    # FIXME - smallest hole redefinition
    smallest_hole_diameter = white_zones[-1]['hole_diameter']
    sampling_interval = source_wavelength * plane_distance / sieve_diameter
    fxx = np.arange(
        -(image_width - 1) / 2,
        (image_width - 1) / 2 + 1
    ) / (image_width * sampling_interval)
    fyy = np.arange(
        -(image_width - 1) / 2,
        (image_width - 1) / 2 + 1
    ) / (image_width * sampling_interval)

    fx, fy = np.meshgrid(fxx, fyy)

    coherent_otf = (
        a(source_wavelength * plane_distance * fx, source_wavelength * plane_distance * fy) *
        np.e**(
            1j * np.pi * (1 / plane_distance + 1 / source_distance) * source_wavelength *
            plane_distance**2 * (fx**2 + fy**2)
        )
    )

    coherent_psf = np.fft.fftshift(np.fft.ifft2(coherent_otf))
    incoherent_psf = np.abs(coherent_psf)**2
    incoherent_otf = np.fft.fftshift(np.fft.fft2(incoherent_psf))

    return incoherent_psf


def generate_measurements(
        source_wavelengths=np.array([33.4, 33.5, 33.6]) * 1e-9,
        measurement_wavelengths=30, sieve_diameter=10e-3,
        smallest_hole_diameter=7.56e-6 * 4, image_width=301, num_copies=10,
        open_area_ratio=0.6, hole_diameter_to_zone_width=1.53096
):
    """
    Generate measurements array for CSBS algorithm

    Args:
        source_wavelengths (ndarray): array of source wavelengths
        measurement_wavelengths (int/ndarray): number of measurement wavelengths, or an array of wavelengths to measure at
        sieve_diameter (float): photon-sieve diameter
        smallest_hole_diameter (float): smallest photon-sieve aperture diameter
        image_width (int): width of psfs (must be odd)

    Returns:
        Measurements object with psfs and csbs data
    """

    image_height = image_width
    num_sources = len(source_wavelengths)

    focal_lengths = sieve_diameter * smallest_hole_diameter / source_wavelengths
    dofs = 2 * smallest_hole_diameter**2 / source_wavelengths
    if type(measurement_wavelengths) is int:
        approx_start = sieve_diameter * smallest_hole_diameter / (max(focal_lengths) + 10 * max(dofs))
        approx_end = sieve_diameter * smallest_hole_diameter / (min(focal_lengths) - 10 * min(dofs))
        measurement_wavelengths = np.linspace(approx_start, approx_end, measurement_wavelengths)

    psfs = np.empty((0, num_sources, image_width, image_width))

    white_zones = photon_sieve(
        sieve_diameter=sieve_diameter,
        smallest_hole_diameter=smallest_hole_diameter,
        hole_diameter_to_zone_width=hole_diameter_to_zone_width,
        open_area_ratio=open_area_ratio
    )

    # generate incoherent measurements for each wavelength and plane location
    for n, measurement_wavelength in enumerate(measurement_wavelengths):

        if n % 10 == 0:
            logging.info('{} iterations'.format(n))

        psf_group = np.empty((0, image_width, image_width))
        for source_wavelength in source_wavelengths:
            psf = incoherent_psf(
                white_zones=white_zones,
                source_wavelength=float(source_wavelength),
                sieve_diameter=float(sieve_diameter),
                smallest_hole_diameter=float(smallest_hole_diameter),
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
    sieve_diameter = 2.5e-2
    smallest_hole_diameter = 5e-6

    #---------- PSF Generation ----------

    psf = incoherent_psf(
        source_wavelength=source_wavelengths[0],
        measurement_wavelength=source_wavelengths[1],
        sieve_diameter=sieve_diameter,
        smallest_hole_diameter=smallest_hole_diameter,
        image_width=image_width
    )

    import matplotlib.pyplot as plt
    plt.imshow(np.abs(psf))
    plt.show()
