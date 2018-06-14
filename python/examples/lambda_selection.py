#!/bin/env python3
# Evan Widloski - 2018-05-22
# plot final csbs result for a range of lambdas
from csbs import csbs
import sse_cost
from psf_generator import generate_measurements
import numpy as np
import pyplot as plt


# run csbs algorithm on a range of lambdas
lambdas = np.logspace(-1, 2, 10)
copies = []
for lam in lambdas:
    print('----------------------', lam)
    measurements = generate_measurements(source_wavelengths=np.array([33.4e-9, 33.7e-9, 33.8e-9]), image_width=51)
    csbs(measurements, sse_cost, 290, lam=lam)
    copies.append(measurements.copies)

# 2D plot of (plane_locations, lambda) vs copies
plt.figure(constrained_layout=True)
plt.imshow(np.abs(copies), cmap='magma', aspect=1)
plt.ylabel('lambda')
plt.xlabel('plane locations')
plt.yticks(np.arange(len(lambdas)), np.round(lambdas, 3))
cbar = plt.colorbar()
cbar.set_label('Copies')
plt.savefig('lambda_selection.png')
