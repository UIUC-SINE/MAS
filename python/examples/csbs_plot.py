#!/bin/env python3
# Evan Widloski - 2018-05-22
# Plot results of CSBS algorithm at each iteration

from mas.csbs import csbs
from mas import sse_cost
from mas.psf_generator import generate_measurements
from mas.plotting import fourier_slices
import numpy as np

# initialize A matrix
measurements = generate_measurements(source_wavelengths=np.array([33.4e-9,  33.7e-9, 33.8e-9]),
                                     image_width=51,
                                     num_copies=10)

# run CSBS (modifies measurements in place)
csbs(measurements, sse_cost, 3, lam=20)

# plot results
plt = fourier_slices(measurements)
plt.show()
# plt.savefig('csbs_fourier_slices.png')
