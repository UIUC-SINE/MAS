#!/bin/env python3
# Evan Widloski - 2018-05-20

import numpy as np
from matplotlib import pyplot as plt


def fourier_slices(measurements):

    psf_ffts = np.fft.fft2(measurements.psfs, axes=(2, 3))

    _, num_sources, image_width, _ = measurements.psfs.shape

    slices = measurements.psfs[:, :, image_width // 2, :]

    fig, subplots = plt.subplots(num_sources + 2, 1)

    for source_index, subplot in enumerate(subplots[:-2]):
        subplot.imshow(slices[:, source_index].T, cmap='magma', interpolation='nearest', aspect='auto')
        # subplot.imshow(slices[:, source_index], cmap='magma')
        subplot.set_title('Source {}'.format(source_index))
        subplot.get_xaxis().set_visible(False)


    copies_progression = np.empty((0, len(measurements.copies)))
    copies = np.ones(len(measurements.copies)) * measurements.num_copies

    for copy_removed in measurements.copies_history:
        copies[copy_removed] -= 1
        copies_progression = np.append(copies_progression, [copies], axis=0)

    # copies_progression = np.flip(copies_progression, axis=1)
    subplots[-2].imshow(copies_progression, cmap='magma', interpolation='nearest', aspect='auto')

    subplots[-1].plot(measurements.plane_locations, measurements.copies)
    subplots[-1].set_xlim([min(measurements.plane_locations), max(measurements.plane_locations)])
    subplots[-1].set_title('Copies')
    subplots[-1].set_xlabel('Plane Location (m)')
    subplots[-1].grid(True)

    # plt.tight_layout()
    plt.show()
