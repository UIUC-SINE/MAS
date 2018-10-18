#!/bin/env python3
# Evan Widloski - 2018-04-18
# functions for generating photon sieve PSFs

import numpy as np
import logging
from decimal import Decimal
import math
import functools

logging.basicConfig(level=logging.INFO, format='%(message)s')


def circ_incoherent_psf(
        *,
        sieve,
        source_wavelength,
        measurement_wavelength,
        image_width,
        sampling_interval,
        **kwargs
):
    """
    Fast approximation of photon sieve PSF using circular aperture

    Args:
        source_wavelength (float): wavelength of monochromatic plane-wave source
        measurement_wavelength (float): wavelength of focal plane to measure at
        sieve (PhotonSieve): a PhotonSieve instance holding sieve parameters
        image_width (int): width of returned psf, in pixels (must be odd)
        sampling_interval (float): size of pizels (m)

    Returns:
        image_width by image_width ndarray containing incoherent psf

    """
    assert image_width % 2 == 1, 'image width must be odd'

    focal_length = sieve.d * sieve.shd / source_wavelength
    plane_location = sieve.d * sieve.shd / measurement_wavelength
    # depth of focus
    dof = 2 * sieve.shd**2 / source_wavelength
    defocus_amount = (plane_location - focal_length) / dof
    # diffraction limited cutoff frequency
    # (equivalent to sieve.d / (source_wavelength * focal_length))
    cutoff_freq = 1 / sieve.shd

    # diffraction limited bandwidth at sensor plane
    diff_limited_bandwidth = sieve.d / (source_wavelength * plane_location)
    # sampling interval in frequency domain
    delta_freq = 1 / (image_width * sampling_interval)

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

def sieve_incoherent_psf(
        *,
        sieve,
        source_wavelength,
        measurement_wavelength,
        image_width,
        source_distance,
        sampling_interval
):
    """
    Generate an incoherent photon-sieve PSF

    Args:
        source_wavelength (float): wavelength of monochromatic plane-wave source
        measurement_wavelength (float): wavelength of focal plane to measure at
        sieve (PhotonSieve): a PhotonSieve instance holding sieve parameters
        image_width (int): width of returned psf, in pixels (must be odd)
        source_distance (float): distance to source
        sampling_interval (float): size of pizels (m)

    Returns:
        image_width by image_width ndarray containing incoherent psf
    """

    assert image_width % 2 == 1, 'image width must be odd'

    def a_func(x, y):
        radius = np.sqrt(x**2 + y**2)
        # return True if point falls inside hole
        for white_zone in sieve.structure:
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

    focal_length = sieve.d * sieve.shd / source_wavelength
    plane_distance = sieve.d * sieve.shd / measurement_wavelength
    # FIXME - smallest hole redefinition
    smallest_hole_diameter = sieve.structure[-1]['hole_diameter']

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


def sieve_structure(sieve):
    """
    Generate photon sieve hole locations and diameters

    Args:
        (PhotonSieve): a PhotonSieve instance holding sieve parameters

    Returns:
        A list of dictionaries representing each 'open' zone with the following keys:

        hole_diameter (float): diameter of holes in this zone
        hole_coordinates (list): list of tuples containing x and y hole coordinates
        inner_radius (float): inner radius of zone accounting for hole overlap
        outer_radius (float): outer radius of zone accounting for hole overlap
    """

    # total number of rings (zones) of holes
    num_white_zones = np.floor(sieve.d**2 / (8 * sieve.d * sieve.shd))
    # radius from sieve center to center of each white zone
    zone_radii = np.sqrt(2 * sieve.d * sieve.shd * np.arange(1, num_white_zones + 1))
    # width of each white zone (outer - inner radius)
    zone_widths = sieve.d * sieve.shd / (2 * zone_radii)
    # diameters of holes in each white zone
    hole_diameters = sieve.hdtzw * zone_widths
    # number of holes in each white zone
    hole_counts = np.round(8 * sieve.oar * zone_widths * zone_radii / (hole_diameters**2))

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


class PhotonSieve():
    """A class for holding photon-sieve parameters and photon-sieve mask

    Args:
        shd (float): smallest photon-sieve hole diameter
        d (float): photon-sieve diameter
        oar (float): open area ratio (ratio of hole area to total sieve area)
        hdtzw (float): ratio of hole diameter to zonewidth

    Attributes:
        shd
        d
        oar
        hdtzw
        structure (list): list of dictionaries repesenting each white_zone
"""

    def __init__(
            self,
            shd=7e-6,
            d=10e-3,
            oar=0.6,
            hdtzw=1.53096,
    ):

        self.shd =shd
        self.d =d
        self.oar = oar 
        self.hdtzw = hdtzw
        self.structure = sieve_structure(self)

class Measurements():
    """A class for holding PSFs and state data during csbs iterations

    Args:
        source_wavelengths (ndarray): array of source wavelengths
        measurement_wavelengths (int/ndarray): an array of wavelengths to measure at.  if an int is given, this array is computed automatically based on source_wavelengths
        image_width (int): width of psfs (must be odd)
        num_copies (int): number of repeated measurements to initialize with
        psf_generator (def): function to generate photon sieve psf (default, mas.psf_generator.sieve_incoherent_psf)
        sieve_generator (def): function to generate sieve structure ( default mas.psf_generator.photon_sieve)

    Attributes:
        psfs (ndarray): an array holding the 2D psfs
        psf_ffts (ndarray): an array holding the 2D psf ffts

        All arguments are also stored as attributes
    """
    def __init__(
            self,
            sieve,
            *,
            source_wavelengths=np.array([33.4, 33.5, 33.6]) * 1e-9,
            measurement_wavelengths=30,
            image_width=301,
            num_copies=10,
            psf_generator=sieve_incoherent_psf,
    ):


        print(sieve)
        print(type(sieve))
        focal_lengths = sieve.d * sieve.shd / source_wavelengths
        dofs = 2 * sieve.shd**2 / source_wavelengths
        if type(measurement_wavelengths) is int:
            approx_start = sieve.d * sieve.shd / (max(focal_lengths) + 10 * max(dofs))
            approx_end = sieve.d * sieve.shd / (min(focal_lengths) - 10 * min(dofs))
            measurement_wavelengths = np.linspace(approx_start, approx_end, measurement_wavelengths)

        psfs = np.empty((0, len(source_wavelengths), image_width, image_width))

        logging.info("Generating psfs...")
        # generate incoherent measurements for each wavelength and plane location
        for m, measurement_wavelength in enumerate(measurement_wavelengths):


            psf_group = np.empty((0, image_width, image_width))
            for n, source_wavelength in enumerate(source_wavelengths):
                logging.info(
                    '{}/{}'.format(
                        m * len(source_wavelengths) + n + 1,
                        len(measurement_wavelengths) * len(source_wavelengths)
                    )
                )
                psf = psf_generator(
                    sieve=sieve,
                    source_wavelength=float(source_wavelength),
                    measurement_wavelength=measurement_wavelength,
                    image_width=image_width,
                    source_distance=float('inf'),
                    sampling_interval=float(sieve.shd)/10
                )

                psf_group = np.append(psf_group, [psf], axis=0)
            psfs = np.append(psfs, [psf_group], axis=0)

        self.psfs = psfs
        self.psf_ffts = np.fft.fft2(psfs)
        self.num_copies = num_copies
        self.copies = np.ones((len(measurement_wavelengths))) * num_copies
        self.measurement_wavelengths = measurement_wavelengths
        self.source_wavelengths = source_wavelengths
        self.image_width = image_width
        self.copies_history = []
