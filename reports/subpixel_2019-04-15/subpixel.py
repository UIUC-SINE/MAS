# Evan Widloski - 2019-04-18
# Subpixel registration test with various levels of poisson noise, circular shift
# based on https://scikit-image.org/docs/dev/auto_examples/transform/plot_register_translation.html

import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from mas.plotting import plotter4d
from mas.forward_model import get_measurements
from mas.psf_generator import PhotonSieve, PSFs
from mas.decorators import vectorize
from scipy.ndimage.interpolation import shift
# from scipy.ndimage import fourier_shift

original = np.load('nanoflare.npy')
offset = (25.33, 25.33)
# shifted = np.fft.ifft2(fourier_shift(np.fft.fft2(original), offset))
shifted = shift(original, shift=offset, mode='wrap')
sources = np.array([original, shifted])

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

measured_original = get_measurements(
    sources=sources[np.newaxis, :, :],
    psfs=psfs,
    real=True
)[0, 0]

measured_shifted = get_measurements(
    sources=shifted[np.newaxis, np.newaxis, :, :],
    psfs=psfs,
    real=True
)[0, 0]

# %% noise -----

max_photon_count_list = [10, 50, 100, 500, 1000]

noisy_list = []
for max_photon_count in max_photon_count_list:

    noisy_original = np.random.poisson(
        max_photon_count * measured_original / np.max(measured_original)
    ) * np.max(measured_original)
    noisy_shifted = np.random.poisson(
        max_photon_count * measured_shifted / np.max(measured_shifted)
    ) * np.max(measured_shifted)
    noisy_list.append([noisy_original, noisy_shifted])

# dimension: (num photon count, 2, X, Y)
noisy_list = np.array(noisy_list)

# %% register -----

offsets = []
errors = []
for noise_level in noisy_list:
    offset, error, diffphase = register_translation(noise_level[0], noise_level[1], 100)
    offsets.append(offset)
    errors.append(error)

correlations = []
for noise_level, offset in zip(noisy_list, offsets):
    image_product = np.fft.fft2(noise_level[0]) * np.fft.fft2(noise_level[1]).conj()
    # preview the correlation map computed by register_translation
    coarse_correlation = np.fft.fftshift(np.fft.ifft2(image_product))
    # preview the correlation map computed by register_translation
    fine_correlation = _upsampled_dft(
        image_product,
        image_product.shape[0],
        100,
        (offset*100) + image_product.shape[0] // 2
    ).conj()
    correlations.append([coarse_correlation, fine_correlation])
correlations = np.array(correlations)


# %% plot -----

plt = plotter4d(
    noisy_list,
    title='Original and shifted images at various noise levels',
    colorbar=False,
    row_labels=max_photon_count_list,
    column_labels=['original', 'shifted'],
    sup_ylabel='max photon count',
    figsize=(4.5, 10),
    cmap='gist_heat'
)
plt.savefig('images.png', dpi=300)

plt = plotter4d(
    np.abs(correlations),
    title='Correlation stages',
    colorbar=False,
    row_labels=max_photon_count_list,
    column_labels=['coarse', 'fine'],
    sup_ylabel='max photon count',
    sup_xlabel='correlations',
    figsize=(4.5, 10),
    cmap='gist_heat'
)
plt.savefig('correlations.png', dpi=300)

print(errors)
