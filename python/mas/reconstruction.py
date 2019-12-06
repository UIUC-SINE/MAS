#!/bin/env python3
# Evan Widloski - 2019-12-05

import numpy as np
from scipy.ndimage import fourier_shift
from mas.misc import shift

# def coadd(images, drift, time_step):
def coadd(images, drift):
    """Form high resolution image from translated images

    Args:
        images: input images
        drift: 2D drift vector
        # time_step: time from beginning of one frame capture to beginning of next
    """

    # pad_amount = np.ceil(np.abs(drift) * time_step * len(images)).astype(int)
    pad_amount = np.ceil(np.abs(drift) * len(images)).astype(int)

    pad_x = (0, pad_amount[0]) if drift[0] > 0 else (pad_amount[0], 0)
    pad_y = (pad_amount[1], 0) if drift[1] > 0 else (0, pad_amount[1])

    padded_images = np.pad(images, ((0, 0), pad_x, pad_y), mode='constant')

    coadded = np.zeros_like(padded_images[0])
    for i, image in enumerate(padded_images):
        coadded += shift(image, (i * drift[0], -i * drift[1]))

    return coadded
