#!/bin/env python3
# Evan Widloski - 2018-05-20

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import figaspect
import logging
from matplotlib.widgets import Slider
from matplotlib.colors import Normalize
import matplotlib.animation as animation

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
        subplot.grid(False)
        # subplot.set(adjustable='box', aspect=1/4)
        subplot.set_title('Source {}'.format(source_index))
        subplot.set_ylabel('Freq.')
        subplot.get_xaxis().set_visible(False)


    # show csbs parameters
    if hasattr(measurements, 'csbs_params'):
        plt.figtext(
            0.98, 0.98, str(measurements.csbs_params),
            horizontalalignment='right',
            rotation='vertical', fontsize='xx-small'
        )
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
        subplots[-1].set_ylabel('Copies')
        subplots[-1].grid(True)

    else:
        logging.warning("No copies_history/copies/csbs_params.  Did you run CSBS?")
        copies_progression = None

    fig.constrained_layout = True

    return plt

def psf_slider(psfs):
    """Plot 1 row of Measurements matrix, with a slider to adjust measurements
    plane

    Args:
        psfs (PSFs): psfs object after CSBS
    """

    n = Normalize()

    fig, subplots = plt.subplots(1, len(psfs.source_wavelengths), squeeze=False)
    subplots = subplots[0]

    ims = [subplot.imshow(psfs.psfs[0, n]) for n, subplot in enumerate(subplots)]

    slider_axis = plt.axes([0.25, 0.05, 0.65, 0.03])
    slider = Slider(
        slider_axis,
        'Measurement plane',
         0,
         len(psfs.measurement_wavelengths) - 1,
         valfmt='%i'
     )

    def update(_):
        measurement_plane_index = int(slider.val)
        print(measurement_plane_index)
        for n, im in enumerate(ims):
            im.set_array(psfs.psfs[measurement_plane_index, n])
            # im.set_norm(Normalize())
        fig.canvas.draw_idle()

    slider.on_changed(update)

    return plt


def slider(images, same_scale=False):
    """Plot array of images with a slider

    Args:
        images (ndarray): (L x M x N) array of (M x N) images
        same_scale (boolean): force colormap to have same scale on all images
    """

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    im = ax.imshow(images[0])

    slider_axis = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(slider_axis, 'Index', 0, len(images) - 1, valfmt='%i')

    global_max = images.max()
    global_min = images.min()

    def update(_):
        im.set_array(images[int(slider.val)])
        im.set_norm(Normalize())
        if same_scale:
            im.set_clim(global_min, global_max)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    return slider


def plotter4d(data, title='', fignum=None, cmap=None, figsize=None,
              colorbar=False, column_labels=None, row_labels=None,
              sup_ylabel=None, sup_xlabel=None, scale=False):
    """Plot 4d ndarrays to the subplots of the first two dimensions

    Args:
        data (ndarray): 4d ndarray to be plotted
        title (string): suptitle of the whole figure
        cmap (string), optional: colormap to use while plotting
        figsize (tuple), optional: figsize to pass to plt.subplots
        column_labels (list of strings), optional: titles on subplot columns
        row_labels (list of strings), optional: titles on subplot rows
        colorbar (boolean), optional: whether to use a colorbar
        scale (boolean), default=False: use same colorscale for all images
    """
    if len(data.shape) == 3:
        data = data[:, np.newaxis, :, :]
    k,p = data.shape[:2]

    if plt.fignum_exists(fignum):
        figo = plt.gcf()
        for i in figo.axes:
            if len(i.images) > 0:
                if i.images[0].colorbar is not None:
                    i.images[0].colorbar.remove()

    fig, subplots = plt.subplots(
        k, p,
        squeeze=False,
        num=fignum,
        figsize=figsize,
        sharex=True,
        sharey=True
    )
    plt.suptitle(title)

    if scale:
        vmin, vmax = np.min(data), np.max(data)
    else:
        vmin, vmax = None, None

    for data_row, subplot_row in zip(data, subplots):
        for data, subplot in zip(data_row, subplot_row):
            im = subplot.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            subplot.axes.get_xaxis().set_ticks([])
            subplot.axes.get_yaxis().set_ticks([])
            if colorbar:
                fig.colorbar(im, ax=subplot)

    if column_labels is not None:
        for subplot, col in zip(subplots[0], column_labels):
            subplot.set_title(col)

    if row_labels is not None:
        for subplot, row in zip(subplots[:, 0], row_labels):
            h = subplot.set_ylabel(row, rotation=0, size='large')
            h.set_rotation(90)

    # hack to add xlabel, ylabel
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    if sup_xlabel is not None:
        plt.xlabel(sup_xlabel)
    if sup_ylabel is not None:
        plt.ylabel(sup_ylabel)

    plt.show()

    return plt


def image_animater(
        arr,
        title_array=None,
        figsize=(7.4, 4.8),
        cmap='gray',
        vmin=None,
        vmax=None,
        title_format='{}',
        outfile=None,
        fps=30,
        same_scale=False
):
    """Create animated sequence from stack of images

    Args:
        arr (ndarray): input stack (num_images, width, height)
        title_array (list): list of values to substitute into title_format
            when generating titles for each image
        figsize (tuple): pyplot figsize
        cmap (str): image colormap
        vmin (float): image scale min
        max (float): image scale max
        title_format (str): format string to generate image titles
        outfile (string): output file path to generate a .gif
        fps (int) framerate of animated sequence
        same_scale (boolean): force images to have same colorscale (default=False)
    """
    plt.ioff()
    if title_array is None:
        title_array = np.arange(arr.shape[0]) + 1
    num = arr.shape[0]
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    im = plt.imshow(arr[0], cmap=cmap)
    cb = fig.colorbar(im, ax=ax)
    global_max = arr.max()
    global_min = arr.min()
    def update(i):
        plt.title(title_format.format(title_array[i]))

        if same_scale:
            im.set_clim(global_min, global_max)

        elif vmin is None:
            im.set_clim(np.min(arr[i]), np.max(arr[i]))

        im.set_data(arr[i])

    anim = animation.FuncAnimation(fig, update, frames=num, interval=1000 // fps)

    if outfile is not None:
        writergif = animation.PillowWriter(fps=fps)
        anim.save(outfile, writergif)

    return anim
