#!/bin/env python3
# Evan Widloski - 2019-04-15
# subpixel tracking test using Prony's method and phase correlation

from imageio import imread
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import face
from scipy.ndimage import rotate
from mas.forward_model import downsample, upsample, zeropad

x = face(gray=True)
roi = upsample(downsample(x[500:600, 500:600], factor=5), factor=5)

x_fft = np.fft.fft2(x)
roi_fft = np.fft.fft2(zeropad(roi, x.shape, mode='topleft'))

shifted_phase = np.abs(
    np.fft.ifft2(
        np.multiply(x_fft, np.conj(roi_fft)) /
        np.abs(np.multiply(x_fft, np.conj(roi_fft)))
    )
)

plt.imshow(shifted_phase)
plt.show(f


