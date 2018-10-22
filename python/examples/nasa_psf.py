#!/bin/env python3
# Evan Widloski - 2018-10-14
# generate some PSFs and save to HDF5 file

from mas.psf_generator import PSFs, PhotonSieve, circ_incoherent_psf
import numpy as np
import h5py

# generate photon sieve
ps = PhotonSieve(
    diameter=80e-3,
    smallest_hole_diameter=7e-6,
    hole_diameter_to_zone_width=7/6
)

# generate psfs for 632.8nm at [0, 1, 2, 3, 5, 7] DOF away from focus
source_wavelengths = np.array([632.8e-9])
dof_array = np.array([0, 1, 2, 3, 5, 7])
dof = 2 * ps.smallest_hole_diameter**2 / source_wavelengths
focal_length = ps.diameter * ps.smallest_hole_diameter / source_wavelengths

psfs = PSFs(
    ps,
    source_wavelengths=source_wavelengths,
    measurement_wavelengths=ps.diameter * ps.smallest_hole_diameter / (focal_length + dof_array * dof),
    image_width=3001,
    sampling_interval=2.2e-6,
    psf_generator=circ_incoherent_psf,
)

# save to file
with h5py.File('/tmp/psfs.h5', 'w') as f:
    for psf, dof in zip(psfs.psfs[:, 0, :, :], dof_array):
        f.create_dataset("dof={}".format(dof), data=psf)
