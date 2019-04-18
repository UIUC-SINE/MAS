#!/bin/env python3
# Evan Widloski - 2019-04-15
# subpixel tracking test using Prony's method and phase correlation

from imageio import imread
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import face
from scipy.ndimage import rotate
from mas.forward_model import downsample, upsample, size_equalizer

offset = (500, 500)
x = face(gray=True)
# roi = upsample(downsample(x[500:600, 500:600], factor=5), factor=5)
crop_roi = x[:-1, :-1]
roll_roi = np.roll(np.roll(x, -offset[0], axis=0), -offset[1], axis=1)

x_fft = np.fft.fft2(x)
# crop_fft = size_equalizer(np.fft.fft2(crop_roi), x.shape)
crop_fft = np.fft.fft2(size_equalizer(crop_roi, x.shape))
roll_fft = np.fft.fft2(roll_roi)

crop_csd = (
        np.multiply(x_fft, np.conj(crop_fft)) /
        np.abs(np.multiply(x_fft, np.conj(crop_fft)))
)
crop_phase = np.abs(np.fft.ifft2(crop_csd))

roll_csd = (
        np.multiply(x_fft, np.conj(roll_fft)) /
        np.abs(np.multiply(x_fft, np.conj(roll_fft)))
)
roll_phase = np.abs(np.fft.ifft2(roll_csd))

def zoom_plot(im, offset, width):
    plt.imshow(np.abs(im)**(3/10))
    plt.xlim([offset[1] - 20, offset[1] + 20])
    plt.ylim([offset[0] - 20, offset[0] + 20])

plt.subplot(3, 2, 1)
plt.imshow(crop_csd.real)
plt.subplot(3, 2, 2)
plt.imshow(roll_csd.real)
plt.subplot(3, 2, 3)
zoom_plot(np.fft.fftshift(crop_fft), np.array(crop_fft.shape) // 2, 40)
plt.subplot(3, 2, 4)
zoom_plot(np.fft.fftshift(roll_fft), np.array(roll_fft.shape) // 2, 40)
plt.subplot(3, 2, 5)
zoom_plot(crop_phase, offset, 40)
plt.subplot(3, 2, 6)
zoom_plot(roll_phase, offset, 40)
plt.show()
