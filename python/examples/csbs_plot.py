#!/bin/env python3
# Evan Widloski - 2018-05-22
# Plot results of CSBS algorithm at each iteration

from mas.csbs import csbs
from mas import sse_cost
from mas.psf_generator import Measurements, PhotonSieve, circ_incoherent_psf
from mas.plotting import fourier_slices
import numpy as np

# create photon sieve with default parameters
ps = PhotonSieve()
# generate psfs
m = Measurements(
    ps,
    source_wavelengths=np.array([32.0e-9,  33.4e-9, 35.0e-9]),
    psf_generator=circ_incoherent_psf,
    image_width=501,
    num_copies=10
)

# run CSBS (modifies measurements in place)
csbs(m, sse_cost, 10, lam=100)

# plot results
plt = fourier_slices(measurements)
plt.show()
# plt.savefig('csbs_fourier_slices.png')
