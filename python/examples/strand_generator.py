# Ulas Kamaci 2019-04-20

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cv2, skimage
def gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

imsize = 512 # initially chose large imsize to prevent pixellation
imsize_final = 160 # then use skimage-resize to get the desired sized image (no pixellation)
numstrands = 100
thickness = 22 # pixel thickness of each strand in the (imsize, imsize) image
# imsize = 160
# numstrands = 100
# thickness = 5 # pixels
max_tilt_angle = 20 # plus minus wrt vertical axis

img = np.zeros((imsize,imsize))
for i in range(numstrands):
    tilt = np.random.randint(-max_tilt_angle, max_tilt_angle) / 180 * np.pi
    x, y = np.random.randint(0,imsize), np.random.randint(0,imsize) # center point of the strand
    pt1 = (int(x + y * np.tan(tilt)), 0) # point where it crosses y=0 line
    pt2 = (int(x - (imsize-y) * np.tan(tilt)), imsize) # point where it crosses y=imsize line
    intensity = np.random.randint(0,100) / 100
    # intensity = np.random.normal(loc=100, scale=10)**2
    # intensity = 1
    im0 = np.zeros((imsize, imsize))
    img += cv2.line(im0, pt1, pt2, intensity, thickness=thickness)

ker = gauss2D(shape=(5,5), sigma=1)
imgf = signal.convolve2d(img, ker, mode='same') # blur the image a little bit
imgr = skimage.transform.resize(imgf, (imsize_final, imsize_final))
imgr = imgr / imgr.max()

plt.figure(2)
plt.imshow(imgr, cmap='gist_heat')
plt.show()
