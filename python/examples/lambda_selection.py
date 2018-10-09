#!/bin/env python3
# Evan Widloski - 2018-05-22
# plot final csbs result for a range of lambdas
from mas.csbs import csbs
from mas import sse_cost
from mas.psf_generator import generate_measurements, circ_incoherent_psf
import numpy as np
from matplotlib import pyplot as plt
import copy


# run csbs algorithm on a range of lambdas
lambdas = np.logspace(-10, 2, 20)
copies = []
orig_measurements = generate_measurements(
    source_wavelengths=np.array([33.4e-9, 33.7e-9, 33.8e-9]),
    measurement_wavelengths=5,
    image_width=51,
    psf_generator=circ_incoherent_psf,
)
for lam in lambdas:
    print('----------------------', lam)
    measurements = copy.deepcopy(orig_measurements)
    csbs(measurements, sse_cost, 10, lam=lam)
    copies.append(measurements.copies)

# 2D plot of (plane_locations, lambda) vs copies
plt.figure(constrained_layout=True)
plt.imshow(np.abs(copies), cmap='magma', aspect=1)
plt.ylabel('lambda')
plt.xlabel('plane locations')
plt.yticks(np.arange(len(lambdas)), np.round(lambdas, 3))
cbar = plt.colorbar()
cbar.set_label('Copies')
# plt.savefig('lambda_selection.png')
