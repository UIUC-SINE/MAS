#!/bin/env python3
# Evan Widloski - 2018-04-18

import numpy as np
import logging
from decimal import Decimal
import math
import functools

logging.basicConfig(level=logging.INFO, format='%(message)s')


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

def circ_incoherent_psf(
        *,
        source_wavelength,
        measurement_wavelength,
        sieve_diameter,
        smallest_hole_diameter,
        image_width,
        sampling_interval,
        **kwargs
):
    """
    Generate an approximation of an incoherent photon-sieve PSF

    Args:
        source_wavelength (float): wavelength of monochromatic plane-wave source
        measurement_wavelength (float): wavelength of focal plane to measure at
        sieve_diameter (float): photon-sieve diameter
        smallest_hole_diameter (float): diameter of smallest hole in photon-sieve
        image_width (int): width of returned psf, in pixels (must be odd)
        sampling_interval (float): size of pizels (m)

    Returns:
        image_width by image_width ndarray containing incoherent psf

    """
    assert image_width % 2 == 1, 'image width must be odd'

    focal_length = sieve_diameter * smallest_hole_diameter / source_wavelength
    plane_location = sieve_diameter * smallest_hole_diameter / measurement_wavelength
    # depth of focus
    dof = 2 * smallest_hole_diameter**2 / source_wavelength
    defocus_amount = (plane_location - focal_length) / dof
    # diffraction limited cutoff frequency
    # (equivalent to sieve_diameter / (source_wavelength * focal_length))
    cutoff_freq = 1 / smallest_hole_diameter

    # diffraction limited bandwidth at sensor plane
    diff_limited_bandwidth = sieve_diameter / (source_wavelength * plane_location)
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
        white_zones,
        source_wavelength,
        measurement_wavelength,
        sieve_diameter,
        smallest_hole_diameter,
        image_width,
        source_distance,
        sampling_interval
):
    """
    Generate an incoherent photon-sieve PSF

    Args:
        white_zones (list): list of dictionaries repesenting each white_zone
        source_wavelength (float): wavelength of monochromatic plane-wave source
        measurement_wavelength (float): wavelength of focal plane to measure at
        sieve_diameter (float): photon-sieve diameter
        smallest_hole_diameter (float): diameter of smallest hole in photon-sieve
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

class Measurements():
    """A class for holding PSFs and state data during csbs iterations

    Args:
        source_wavelengths (ndarray): array of source wavelengths
        measurement_wavelengths (int/ndarray): an array of wavelengths to measure at.  if an int is given, this array is computed automatically based on source_wavelengths
        sieve_diameter (float): photon-sieve diameter
        smallest_hole_diameter (float): smallest photon-sieve aperture diameter
        image_width (int): width of psfs (must be odd)
        num_copies (int): number of repeated measurements to initialize with
        psf_generator (def): function to generate photon sieve psf (default, mas.psf_generator.sieve_incoherent_psf)
        sieve_generator (def): function to generate sieve structure ( default mas.psf_generator.photon_sieve)

    Attributes:
        psfs (ndarray): an array holding the 2D psfs
        psf_dfts (ndarray): an array holding the 2D psf dfts

        All arguments are also stored as attributes
    """
    def __init__(
            self,
            source_wavelengths=np.array([33.4, 33.5, 33.6]) * 1e-9,
            measurement_wavelengths=30,
            sieve_diameter=10e-3,
            image_width=301,
            open_area_ratio=0.6,
            hole_diameter_to_zone_width=1.53096,
            psf_generator=sieve_incoherent_psf,
            sieve_generator=photon_sieve,
            smallest_hole_diameter=7e-6
    ):


        focal_lengths = sieve_diameter * smallest_hole_diameter / source_wavelengths
        dofs = 2 * smallest_hole_diameter**2 / source_wavelengths

        # if measurement_wavelengths was an int compute a uniform range of measurement_wavelengths
        if type(measurement_wavelengths) is int:
            approx_start = sieve_diameter * smallest_hole_diameter / (max(focal_lengths) + 10 * max(dofs))
            approx_end = sieve_diameter * smallest_hole_diameter / (min(focal_lengths) - 10 * min(dofs))
            measurement_wavelengths = np.linspace(approx_start, approx_end, measurement_wavelengths)

        self.psfs = np.empty((0, len(source_wavelengths), image_width, image_width))

        logging.info("Generating photon sieve structure...")
        # generate photon sieve structure
        white_zones = photon_sieve(
            sieve_diameter=sieve_diameter,
            smallest_hole_diameter=smallest_hole_diameter,
            hole_diameter_to_zone_width=hole_diameter_to_zone_width,
            open_area_ratio=open_area_ratio
        )

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
                    white_zones=white_zones,
                    source_wavelength=float(source_wavelength),
                    sieve_diameter=float(sieve_diameter),
                    smallest_hole_diameter=float(smallest_hole_diameter),
                    measurement_wavelength=measurement_wavelength,
                    image_width=image_width,
                    source_distance=float('inf'),
                    sampling_interval=float(smallest_hole_diameter)/10
                )

                psf_group = np.append(psf_group, [psf], axis=0)
            self.psfs = np.append(self.psfs, [psf_group], axis=0)

        self.psf_dfts = np.fft.fft2(self.psfs)
        self.source_wavelengths = source_wavelengths
        self.measurement_wavelengths = measurement_wavelengths
        self.sieve_diameter = sieve_diameter
        self.image_width = image_width
        self.open_area_ratio = open_area_ratio
        self.hole_diameter_to_zone_width = hole_diameter_to_zone_width
        self.psf_generator = psf_generator
        self.sieve_generator = sieve_generator
        self.smallest_hole_diameter = smallest_hole_diameter



        # self.psfs = psfs
        # self.psf_ffts = np.fft.fft2(psfs)
        # self.num_copies = num_copies
        # self.copies = np.ones((len(measurement_wavelengths))) * num_copies
        # self.measurement_wavelengths = measurement_wavelengths
        # self.image_width = psfs.shape[2]
        # self.copies_history = []
