#!/bin/env python3
# Evan Widloski - 2018-05-20

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import figaspect
import logging
from matplotlib.widgets import Slider
from matplotlib.colors import Normalize

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

    # calculate PSF dfts and slice each at x=0
    slices = np.fft.fftshift(measurements.psf_dfts, axes=(2, 3))[:, :, image_width // 2, :]

    # plot frequency support of each PSF at measurement planes
    fig, subplots = plt.subplots(num_sources + 2, 1,
                                 constrained_layout=True,
                                 figsize=figaspect(2),
    )
    for source_index, subplot in enumerate(subplots[:-2]):
        subplot.imshow(np.abs(slices[:, source_index].T), cmap='magma', interpolation='nearest', aspect='auto')
        # subplot.set(adjustable='box', aspect=1/4)
        subplot.set_title('Source {}'.format(source_index))
        subplot.set_ylabel('Frequency support')
        subplot.get_xaxis().set_visible(False)


    # show csbs parameters
    if hasattr(measurements, 'csbs_params'):
        plt.figtext(0.98, 0.98, str(measurements.csbs_params),
                    horizontalalignment='right',
                    rotation='vertical')
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

        # plot final copies
        subplots[-1].plot(measurements.measurement_wavelengths, measurements.copies, 'o')
        subplots[-1].set_xlim(
            [min(measurements.measurement_wavelengths),
            max(measurements.measurement_wavelengths)]
        )
        subplots[-1].set_title('Final Iteration')
        subplots[-1].set_xlabel('Plane Location (m)')
        subplots[-1].grid(True)

    else:
        logging.warning("No copies_history/copies/csbs_params.  Did you run CSBS?")

    fig.constrained_layout = True

    return plt

def psf_slider(measurements):
    """Plot 1 row of Measurements matrix, with a slider to adjust measurements
    plane

    Args:
        measurements (Measurements): measurements object after CSBS
    """

    n = Normalize()

    fig, subplots = plt.subplots(1, len(measurements.source_wavelengths), squeeze=False)
    subplots = subplots[0]

    ims = [subplot.imshow(measurements.psfs[0, n]) for n, subplot in enumerate(subplots)]

    slider_axis = plt.axes([0.25, 0.05, 0.65, 0.03])
    slider = Slider(
        slider_axis,
        'Measurement plane',
         0,
         len(measurements.measurement_wavelengths) - 1,
         valfmt='%i'
     )

    def update(_):
        measurement_plane_index = int(slider.val)
        print(measurement_plane_index)
        for n, im in enumerate(ims):
            im.set_array(measurements.psfs[measurement_plane_index, n])
            im.set_norm(Normalize())
        fig.canvas.draw_idle()

    slider.on_changed(update)

    return plt
