import numpy as np
from scipy import misc
import scipy.stats as st
from matplotlib import pyplot as plt
from skimage.transform import resize
from computational_functions import *

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

block_dim = 31
fft_dim = 51

'''h = gauss2D(shape=(block_dim,block_dim),sigma=0.5)
H = block_fft2(h,block_dim,fft_dim)

fig = plt.figure()
plt.subplot(121)
plt.imshow(h)
plt.subplot(122)
plt.imshow(abs(H))
plt.show()'''



x1 = misc.imread('SI1_1.jpg', flatten='true')/255
x2 = misc.imread('SI1_2.jpg', flatten='true')/255
print('x1_energy = %d ' % sum(sum(x1*x1)))
print('x2_energy = %d ' % sum(sum(x2*x2)))

# fig = plt.figure()
# plt.subplot(121)
# plt.imshow(x1, cmap='gray')
# plt.subplot(122)
# plt.imshow(x2, cmap='gray')
# plt.show()

x = np.concatenate((x1,x2),axis=0)
fig = plt.figure()
plt.imshow(x, cmap='gray')
plt.show()

imsize = x1.shape[0]
print('imsize=%d' % imsize)

h11 = gauss2D(shape=(block_dim,block_dim),sigma=1)
h12 = gauss2D(shape=(block_dim,block_dim),sigma=20)
h21 = gauss2D(shape=(block_dim,block_dim),sigma=20)
h22 = gauss2D(shape=(block_dim,block_dim),sigma=1)

psf_mtx = np.concatenate((np.concatenate((h11,h12),axis=1),np.concatenate((h21,h22),axis=1)),axis=0)
otf_mtx = block_fft2(psf_mtx,block_dim,imsize)

fig = plt.figure()
plt.subplot(121)
plt.imshow(psf_mtx)
plt.subplot(122)
plt.imshow(abs(otf_mtx))
plt.show()

Fx = block_fft2(x,imsize,imsize)
fig = plt.figure()
plt.imshow(abs(Fx)**0.2, cmap='gray')
plt.show()

y = block_ifft2(my_mul(otf_mtx,Fx,imsize),imsize)

fig = plt.figure()
plt.imshow(abs(y), cmap='gray')
plt.show()

'''numblocks = 100
block_dim = 128

a = np.random.random((block_dim*numblocks,block_dim*numblocks))
ah = my_herm(a,block_dim)
aha = my_mul(ah,a,block_dim)
b = my_inv(aha,block_dim)
c = my_mul(aha,b,block_dim)

plt.figure()
plt.imshow(aha)
plt.show()

plt.figure()
plt.imshow(c)
plt.show()'''
