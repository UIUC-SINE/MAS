# simulate a noisy 20s observation from Klimchuk's simulation

# %% load

import os
import numpy as np
import h5py
from matplotlib.animation import ArtistAnimation
from mas.psf_generator import PSFs, PhotonSieve, circ_incoherent_psf
from mas.forward_model import get_measurements
import matplotlib.pyplot as plt
from mas.plotting import plotter4d

path = os.path.expanduser('~/documents/mas/nanoflare_videos/NanoMovie0_2000strands_94.h5')
nanoflare_list = h5py.File(path)['NanoMovie0_2000strands_94'][0:20]

# %% psfs -----

ps = PhotonSieve(diameter=10e-2)
wavelengths = np.array([33.4e-9])
psfs = PSFs(
    sieve=ps,
    source_wavelengths=wavelengths,
    measurement_wavelengths=wavelengths,
    num_copies=1
)

# %% measure -----

measured_nanoflare_list = []
for nanoflare in nanoflare_list:
    measured_nanoflare_list.append(
        get_measurements(
            sources=nanoflare[np.newaxis, np.newaxis, :, :],
            psfs=psfs,
            real=True
        )[0, 0]
    )

# %% noise -----

max_photon_count_list = [10, 50, 100, 500, 1000]

noisy_nanoflares = []
for max_photon_count in max_photon_count_list:

    noisy_nanoflare_list = []
    for nanoflare in measured_nanoflare_list:
        noisy_nanoflare = np.random.poisson(
            max_photon_count * nanoflare / np.max(nanoflare)
        ) * np.max(nanoflare)
        noisy_nanoflare_list.append(noisy_nanoflare)

    noisy_nanoflares.append(noisy_nanoflare_list)

# dimension: (num photon count, num measurements, X, Y)
noisy_nanoflares = np.array(noisy_nanoflares)

# %% plot -----

# single snapshot images

plt = plotter4d(
    noisy_nanoflares[:, [0, -1]],
    title='Noisy, single-snapshot measurements at ',
    colorbar=False,
    row_labels=max_photon_count_list,
    column_labels=['t=0s', 't=19s'],
    sup_ylabel='Max photon count',
    figsize=(4.5, 10),
    cmap='gist_heat'
)
plt.savefig('first_last_comparison.png', dpi=300)

plt.figure()
plt.imshow(measured_nanoflare_list[0], cmap='gist_heat')
plt.axis('off')
plt.title('Blurred, single-snapshot measurement')
plt.savefig('single_measured.png', dpi=300)

plt.figure()
plt.imshow(nanoflare_list[0], cmap='gist_heat')
plt.axis('off')
plt.title('Single-snapshot measurement')
plt.savefig('single.png', dpi=300)
