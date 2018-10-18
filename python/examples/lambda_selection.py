#!/bin/env python3
# Evan Widloski - 2018-05-22
# plot final csbs result for a range of lambdas
from mas.csbs import csbs
from mas import sse_cost
from mas.psf_generator import Measurements, PhotonSieve, circ_incoherent_psf
import numpy as np
from matplotlib import pyplot as plt
import copy


# generate photon sieve with default parameters
ps = PhotonSieve()

# run csbs algorithm on a range of lambdas
lambdas = np.logspace(-10, 2, 20)
copies = []
orig_m = Measurements(
    ps,
    source_wavelengths=np.array([33.4e-9, 33.7e-9, 33.8e-9]),
    psf_generator=circ_incoherent_psf,
    image_width=51
)
for lam in lambdas:
    print('----------------------', lam)
    m = copy.deepcopy(orig_m)
    csbs(m, sse_cost, 10, lam=lam)
    copies.append(m.copies)

# 2D plot of (plane_locations, lambda) vs copies
plt.figure(constrained_layout=True)
plt.imshow(np.abs(copies), cmap='magma', aspect=1)
plt.ylabel('lambda')
plt.xlabel('plane locations')
plt.yticks(np.arange(len(lambdas)), np.round(lambdas, 3))
cbar = plt.colorbar()
cbar.set_label('Copies')
# plt.savefig('lambda_selection.png')
