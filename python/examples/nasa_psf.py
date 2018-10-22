#!/bin/env python3
# Evan Widloski - 2018-10-14
# Generate PSFs for Doug

from mas.psf_generator import Measurements, PhotonSieve, sieve_incoherent_psf, circ_incoherent_psf
from mas.plotting import psf_slider
import numpy as np

# ----- parameters -----
wavelength = np.array([632.8e-9])
sieve_diameter=80e-3
smallest_hole_diameter=7e-6
outer_zone_width=6e-6

dof = 2 * smallest_hole_diameter**2 / wavelength
# dof_array = np.arange(-20,21)
dof_array = [0, 5]
focal_length = sieve_diameter * smallest_hole_diameter / wavelength

ps = PhotonSieve(shd=smallest_hole_diameter,
    d=sieve_diameter,
    hdtzw=smallest_hole_diameter/outer_zone_width
    # generate_mask=True,
    # mask_width=1001
)

m = Measurements(
    ps,
    source_wavelengths=wavelength,
    measurement_wavelengths=sieve_diameter * smallest_hole_diameter / (focal_length + dof_array * dof),
    psf_generator=sieve_incoherent_psf,
    image_width=1001,
    sampling_interval=2.2e-6
)

plt = psf_slider(m)
plt.show()
