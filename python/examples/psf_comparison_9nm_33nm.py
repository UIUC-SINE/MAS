#!/bin/env python3
# Comparison of in focus and out of focus PSFs for 9nm and 33.4nm

import matplotlib.pyplot as plt
from mas.psf_generator import PSFs, PhotonSieve, circ_incoherent_psf, sieve_incoherent_psf
from mas.plotting import plotter4d
import numpy as np

wavelengths = np.array([9, 33.5]) * 1e-9
ps = PhotonSieve(diameter=8e-2)
psfs = PSFs(
    sieve=ps,
    source_wavelengths=wavelengths,
    measurement_wavelengths=wavelengths,
    image_width=301,
    psf_generator=sieve_incoherent_psf
)

# focused 9nm
plt.imshow(np.abs(psfs.psfs[0, 0]))
plt.axis([140, 160, 140, 160])
plt.title('9 nm, focused')
plt.savefig('9nm.png')
# focused 33.5nm
plt.imshow(np.abs(psfs.psfs[1, 1]))
plt.axis([140, 160, 140, 160])
plt.title('33.4 nm, focused')
plt.savefig('33.4nm.png')
plt.show()
