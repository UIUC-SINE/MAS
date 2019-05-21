#!/bin/env python3
# Evan Widloski - 2018-04-18
# functions for generating photon sieve PSFs

import numpy as np
from decimal import Decimal
import math
import functools
import sys

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

    focal_length = sieve.diameter * sieve.smallest_hole_diameter / source_wavelength
    plane_location = sieve.diameter * sieve.smallest_hole_diameter / measurement_wavelength
    # depth of focus
    dof = 2 * sieve.smallest_hole_diameter**2 / source_wavelength
    defocus_amount = (plane_location - focal_length) / dof
    # diffraction limited cutoff frequency
    # (equivalent to sieve.diameter / (source_wavelength * focal_length))
    cutoff_freq = 1 / sieve.smallest_hole_diameter

    # diffraction limited bandwidth at sensor plane
    diff_limited_bandwidth = sieve.diameter / (source_wavelength * plane_location)
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

    focal_length = sieve.diameter * sieve.smallest_hole_diameter / source_wavelength
    plane_distance = sieve.diameter * sieve.smallest_hole_diameter / measurement_wavelength

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
        sieve.mask(source_wavelength * plane_distance * fx, source_wavelength * plane_distance * fy) *
        np.e**(
            1j * np.pi * (1 / plane_distance + 1 / source_distance) * source_wavelength *
            plane_distance**2 * (fx**2 + fy**2)
        )
    )

    coherent_psf = np.fft.fftshift(np.fft.ifft2(coherent_otf))
    incoherent_psf = np.abs(coherent_psf)**2

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
    num_white_zones = np.floor(sieve.diameter**2 / (8 * sieve.diameter * sieve.smallest_hole_diameter))
    # radius from sieve center to center of each white zone
    zone_radii = np.sqrt(2 * sieve.diameter * sieve.smallest_hole_diameter * np.arange(1, num_white_zones + 1))
    # width of each white zone (outer - inner radius)
    zone_widths = sieve.diameter * sieve.smallest_hole_diameter / (2 * zone_radii)
    # diameters of holes in each white zone
    hole_diameters = sieve.hole_diameter_to_zone_width * zone_widths
    # number of holes in each white zone
    hole_counts = np.round(8 * sieve.open_area_ratio * zone_widths * zone_radii / (hole_diameters**2))

    white_zones = []
    # generate each white zone
    for hole_diameter, hole_count, zone_radius in zip(hole_diameters, hole_counts, zone_radii):
        white_zone = {}
        white_zone['hole_diameter'] = hole_diameter
        white_zone['hole_coordinates'] = []
        white_zone['inner_radius'] = zone_radius - hole_diameter / 2
        white_zone['outer_radius'] = zone_radius + hole_diameter / 2
        # generate each hole
        theta = 2 * np.pi * np.arange(hole_count) / hole_count
        white_zone['hole_coordinates'] = list(
            zip(
                zone_radius * np.cos(theta),
                zone_radius * np.sin(theta),
            )
        )
        white_zones.append(white_zone)

    return white_zones


class PhotonSieve():
    """A class for holding photon-sieve parameters and photon-sieve mask

    Args:
        smallest_hole_diameter (float): smallest photon-sieve hole diameter
        diameter (float): photon-sieve diameter
        open_area_ratio (float): open area ratio (ratio of hole area to total sieve area)
        hole_diameter_to_zone_width (float): ratio of hole diameter to zonewidth

    Attributes:
        smallest_hole_diameter
        diameter
        open_area_ratio
        hole_diameter_to_zone_width
        structure (list): list of dictionaries repesenting each white_zone
        mask (function): vectorized function that outputs the binary sieve mask a(x,y)
"""

    def __init__(
            self,
            smallest_hole_diameter=7e-6,
            diameter=16e-2,
            open_area_ratio=0.6,
            hole_diameter_to_zone_width=1.53096
    ):

        self.smallest_hole_diameter = smallest_hole_diameter
        self.diameter = diameter
        self.open_area_ratio = open_area_ratio
        self.hole_diameter_to_zone_width = hole_diameter_to_zone_width


        def mask(self, x, y):
            if not hasattr(self, 'structure'):
                self.structure = sieve_structure(self)

            outer_radii = [zone['outer_radius'] for zone in self.structure]
            inner = self.structure[0]['inner_radius']
            outer = self.structure[-1]['outer_radius']

            radius = np.sqrt(x**2 + y**2)
            # return true if point falls inside hole
            if radius >= inner and radius < outer:
                white_zone = self.structure[np.searchsorted(outer_radii, radius)]
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

        self.mask = np.vectorize(mask)

    def get_mask(self, mask_width):
        """
        Return array representing the mask

        Args:
            mask_width (int): number of samples in each direction

        Returns:
            2d ndarray of the generated binary mask
        """

        xx = self.diameter * np.arange(
            -(mask_width - 1) / 2,
            (mask_width - 1) / 2 + 1
        ) / mask_width
        yy = xx
        x, y = np.meshgrid(xx, yy)
        return self.mask(x,y)


class PSFs():
    """A class for holding PSFs and state data during csbs iterations

    Args:
        source_wavelengths (ndarray): array of source wavelengths
        measurement_wavelengths (int/ndarray): an array of wavelengths to measure at.  if an int is given, this array is computed automatically based on source_wavelengths
        image_width (int): width of psfs (must be odd)
        num_copies (int): number of repeated measurements to initialize with
        psf_generator (def): function to generate photon sieve psf (default, mas.psf_generator.sieve_incoherent_psf)

    Attributes:
        psfs (ndarray): an array holding the 2D psfs
        psf_dfts (ndarray): an array holding the 2D psf ffts

        All arguments are also stored as attributes
    """
    def __init__(
            self,
            sieve,
            *,
            source_wavelengths=np.array([33.4, 33.5]) * 1e-9,
            measurement_wavelengths=30,
            image_width=301,
            num_copies=10,
            psf_generator=circ_incoherent_psf,
            sampling_interval=3.5e-6
    ):


        focal_lengths = sieve.diameter * sieve.smallest_hole_diameter / source_wavelengths
        dofs = 2 * sieve.smallest_hole_diameter**2 / source_wavelengths
        if type(measurement_wavelengths) is int:
            approx_start = sieve.diameter * sieve.smallest_hole_diameter / (max(focal_lengths) + 10 * max(dofs))
            approx_end = sieve.diameter * sieve.smallest_hole_diameter / (min(focal_lengths) - 10 * min(dofs))
            measurement_wavelengths = np.linspace(approx_start, approx_end, measurement_wavelengths)
            measurement_wavlengths = np.insert(
                measurement_wavelengths,
                np.searchsorted(measurement_wavelengths, source_wavelengths),
                source_wavelengths
            )

        psfs = np.empty((0, len(source_wavelengths), image_width, image_width))

        # generate incoherent measurements for each wavelength and plane location
        for m, measurement_wavelength in enumerate(measurement_wavelengths):


            psf_group = np.empty((0, image_width, image_width))
            for n, source_wavelength in enumerate(source_wavelengths):
                sys.stdout.write('\033[K')
                print(
                    'PSF {}/{}\r'.format(
                        m * len(source_wavelengths) + n + 1,
                        len(measurement_wavelengths) * len(source_wavelengths)
                    ),
                    end=''
                )
                psf = psf_generator(
                    sieve=sieve,
                    source_wavelength=float(source_wavelength),
                    measurement_wavelength=measurement_wavelength,
                    image_width=image_width,
                    source_distance=float('inf'),
                    sampling_interval=float(sampling_interval)
                )

                psf_group = np.append(psf_group, [psf], axis=0)
            psfs = np.append(psfs, [psf_group], axis=0)

        self.psfs = psfs
        self.psf_dfts = np.fft.fft2(psfs)
        self.num_copies = num_copies
        self.copies = np.ones((len(measurement_wavelengths))) * num_copies
        self.measurement_wavelengths = measurement_wavelengths
        self.source_wavelengths = source_wavelengths
        self.sampling_interval = sampling_interval
        self.image_width = image_width
        self.copies_history = []
