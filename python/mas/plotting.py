#!/bin/env python3
# Evan Widloski - 2018-05-20

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import figaspect
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def fourier_slices(measurements):
    """Plot results of CSBS algorithm
    Plots the frequency support at x=0 for the PSFs at all measurement planes
    Plots the number of measurements taken (copies) at each measurement plane vs iterations
    Plots the final number of copies at the last iteration at each measurement plane

    Args:
        measurements (Measurements): measurements object after CSBS
    """

    _, num_sources, image_width, _ = measurements.psfs.shape

    # calculate PSF ffts and slice each at x=0
    slices = np.fft.fftshift(measurements.psf_ffts, axes=(2, 3))[:, :, image_width // 2, :]

    # plot frequency support of each PSF at measurement planes
    fig, subplots = plt.subplots(num_sources + 2, 1,
                                 constrained_layout=True,
                                 figsize=figaspect(2),
    )
    for source_index, subplot in enumerate(subplots[:-2]):
        subplot.imshow(np.abs(slices[:, source_index].T)**(3/10), cmap='magma', interpolation='nearest', aspect='auto')
        # subplot.set(adjustable='box', aspect=1/4)
        subplot.set_title('Source {}'.format(source_index))
        subplot.set_ylabel('Frequency support')
        subplot.get_xaxis().set_visible(False)


    if measurements.copies_history:
        # plot copies at each plane vs iterations
        copies_progression = np.empty((0, len(measurements.copies)))
        copies = np.ones(len(measurements.copies)) * measurements.num_copies
        for copy_removed in measurements.copies_history:
            copies[copy_removed] -= 1
            copies_progression = np.append(copies_progression, [copies], axis=0)
        img = subplots[-2].imshow(copies_progression, cmap='magma', interpolation='nearest', aspect='auto')
        subplots[-2].set_ylabel('Iterations')
        subplots[-2].get_xaxis().set_visible(False)
        cbar = plt.colorbar(img, ax=subplots[-2])
        cbar.set_label('Copies')
    else:
        logging.warning("No copies_history.  Did you run CSBS?")

    # plot final copies
    subplots[-1].plot(measurements.measurement_wavelengths, measurements.copies, 'o')
    subplots[-1].set_xlim(
        [min(measurements.measurement_wavelengths),
         max(measurements.measurement_wavelengths)]
    )
    subplots[-1].set_title('Final Iteration')
    subplots[-1].set_xlabel('Plane Location (m)')
    subplots[-1].grid(True)

    # show csbs parameters
    if hasattr(measurements, 'csbs_params'):
        plt.figtext(0.98, 0.98, str(measurements.csbs_params),
                    horizontalalignment='right',
                    rotation='vertical')

    fig.constrained_layout = True

    return plt
