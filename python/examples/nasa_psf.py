#!/bin/env python3
# Evan Widloski - 2018-10-14
# Generate PSFs for Doug

from mas.psf_generator import Measurements, PhotonSieve, circ_incoherent_psf
import numpy as np

# generate photon sieve
sieve_diameter=80e-3
smallest_hole_diameter=7e-6
outer_zone_width=6e-6

ps = PhotonSieve(
    d=sieve_diameter,
    shd=smallest_hole_diameter,
    hdtzw=smallest_hole_diameter/outer_zone_width
)

wavelength = np.array([632.8e-9])

dof = 2 * smallest_hole_diameter**2 / wavelength
dof_array = np.array([0, 1, 2, 3, 5, 7])
focal_length = sieve_diameter * smallest_hole_diameter / wavelength

measurements = Measurements(
    ps,
    source_wavelengths=wavelength,
    measurement_wavelengths=sieve_diameter * smallest_hole_diameter / (focal_length + dof_array * dof),
    image_width=3001,
    psf_generator=circ_incoherent_psf,
)

with h5py.File('~/tmp/psfs.h5') as f:
    for psf, dof in zip(measurements.psfs[:, 0, :, :], dof_array):
        f.create_dataset("dof={}".format(dof), data=psf)
