#!/bin/env python3
# Evan Widloski - 2018-10-14
# Generate PSFs for Doug

from mas.psf_generator import Measurements, sieve_incoherent_psf, photon_sieve
import numpy as np

# ----- parameters -----
wavelength = np.array([632.8e-9])
sieve_diameter=80e-3
smallest_hole_diameter=7e-6
outer_zone_width=6e-6

dof = 2 * smallest_hole_diameter**2 / wavelength
focal_length = sieve_diameter * smallest_hole_diameter / wavelength

measurements = Measurements(
    source_wavelengths=wavelength,
    measurement_wavelengths=sieve_diameter * smallest_hole_diameter / (focal_length + 5 * dof),
    sieve_diameter=80e-3,
    hole_diameter_to_zone_width=smallest_hole_diameter/outer_zone_width,
    smallest_hole_diameter=smallest_hole_diameter,
    sieve_generator=photon_sieve,
    psf_generator=sieve_incoherent_psf,
    image_width=3001
)
