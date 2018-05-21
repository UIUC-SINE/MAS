#!/bin/env python3
# Evan Widloski - 2018-05-20

import numpy as np
from matplotlib import pyplot as plt


def fourier_slices(measurements):

    psf_ffts = np.fft.fft2(measurements['psfs'], axes=(2, 3))

    _, num_sources, image_width, _ = measurements['psfs'].shape

    slices = measurements['psfs'][:, :, image_width // 2, :]

    fig, subplots = plt.subplots(num_sources + 1, 1)

    for source_index, subplot in enumerate(subplots[:-1]):
        subplot.imshow(slices[:, source_index].T, interpolation='nearest', aspect='auto')
        subplot.set_title('Source {}'.format(source_index))
        subplot.get_xaxis().set_visible(False)

    subplots[-1].plot(measurements['wavelengths'], measurements['num_copies'])
    subplots[-1].set_title('Copies')
    subplots[-1].set_xlabel('Frequency (Hz)')
    subplots[-1].grid(True)

    plt.tight_layout()
    plt.show()
