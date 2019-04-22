# Ulas Kamaci 2019-04-20

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage.filters import gaussian_filter
import cv2, skimage

imsize = 512 # initially chose large imsize to prevent pixellation
imsize_final = 160 # then use skimage-resize to get the desired sized image (no pixellation)
numstrands = 100
thickness = 22 # pixel thickness of each strand in the (imsize, imsize) image
max_tilt_angle = 20 # plus minus wrt vertical axis

img = np.zeros((imsize,imsize))
for i in range(numstrands):
    theta = np.random.uniform(-max_tilt_angle, max_tilt_angle) / 180 * np.pi
    x  = np.random.randint(
        0 if theta > 0 else imsize * np.tan(theta),
        imsize + imsize * np.tan(theta) if theta > 0 else imsize
    )

    pt1 = (x - int(imsize * np.tan(theta)), 0) # point where it crosses y=0 line
    pt2 = (x, imsize) # point where it crosses y=imsize line
    intensity = np.random.rand()

    im0 = np.zeros((imsize, imsize))
    img += cv2.line(im0, pt1, pt2, intensity, thickness=thickness)

# blurred = gaussian_filter(img, sigma=1)
imgr = skimage.transform.resize(img, (imsize_final, imsize_final))
imgr = imgr / imgr.max()

plt.subplot(2, 1, 1)
plt.imshow(img, cmap='gist_heat')
plt.subplot(2, 1, 2)
plt.imshow(imgr, cmap='gist_heat')
plt.show()
