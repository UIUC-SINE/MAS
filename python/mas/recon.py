import mas
from mas.psf_generator import generate_measurements, circ_incoherent_psf
from mas.sse_cost import block_mul, block_inv, block_herm, block_inv2, block_inv3
from mas.csbs import csbs
from mas import sse_cost
import numpy as np
import scipy.misc
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from mas.pssi_fwd_model import add_noise, size_equalizer
from matplotlib import pyplot as plt

p = 2
order = 1
lam = 1e-1
wavelengths = np.array([33.4e-9, 34e-9])

measurements = generate_measurements(measurement_wavelengths=wavelengths, num_copies=1,
    psf_generator = mas.psf_generator.circ_incoherent_psf,
    source_wavelengths=wavelengths)

# csbs(measurements, sse_cost, 5, lam=lam)
# import ipdb; ipdb.set_trace()

sources = np.array([scipy.misc.face(gray=True)]*p)[:,np.newaxis,:512,:512]/250
aa, bb = sources.shape[2:]
num_sources = sources.shape[0]
sources[1,0,:,:] = np.load('/home/kamo/Projects/mas/python/mas/image_sets/set2/w193_512c_v11.npy')[:aa,:bb]/128


measurement_psfs = np.zeros((num_sources,p,aa,bb))
# reshape psfs
for i in range(num_sources):
    for j in range(p):
        measurement_psfs[i,j,:,:] = size_equalizer(measurements.psfs[i,j,:,:], [aa,bb])

measurement_psf_dfts = fft2(ifftshift(measurement_psfs, axes=(2, 3)))

measurement_psf_dfts_h = block_herm(measurement_psf_dfts)
GAM = block_mul(measurement_psf_dfts_h, measurement_psf_dfts)
if order is 0:
    LAM = np.ones((aa,bb))
elif order is 1:
    difx_ker = np.zeros((aa,bb))
    dify_ker = np.zeros((aa,bb))
    difx_ker[0,0] = -1 ; difx_ker[0,1] = 1
    dify_ker[0,0] = -1 ; dify_ker[1,0] = 1
    LAM = abs(fft2(difx_ker))**2 + abs(fft2(dify_ker))**2
elif order is 2:
    difx_ker = np.zeros((aa,bb))
    dify_ker = np.zeros((aa,bb))
    difx_ker[0,0] = 1 ; difx_ker[0,1] = -2 ; difx_ker[0,2] = 1
    dify_ker[0,0] = 1 ; dify_ker[1,0] = -2 ; difx_ker[2,0] = 1
    LAM = abs(fft2(difx_ker))**2 + abs(fft2(dify_ker))**2

# ----- forward -----
measured = fftshift(
    ifft2(
        block_mul(
            measurement_psf_dfts,
            fft2(ifftshift(sources,axes=(2,3)))
                )
        ), axes=(2,3)
)

measured = np.real(measured)
measured_noisy = np.zeros(measured.shape)

for i in range(measured.shape[0]):
    measured_noisy[i,0,:,:] = add_noise(signal = measured[i,0,:,:], snr_dB = 10)

# FIXME
# measured_noisy = measured

LAM2 = np.einsum('ij,kl', np.eye(num_sources), LAM )
# SIG = GAM
SIG = GAM+lam*LAM2
SIG_inv = block_inv(SIG)
adj_im = block_mul(
    measurement_psf_dfts_h,
    fft2(ifftshift(measured_noisy,axes=(2,3)))
)

recon = fftshift(
    ifft2(
        block_mul( SIG_inv, adj_im )
    ), axes=(2,3)
)

band = 0
fig, ax = plt.subplots(nrows = 2, ncols = 3)
# im0 = ax[0].imshow(sources[1,0,250:700,550:770])
im0 = ax[0,0].imshow(sources[0][0])
fig.colorbar(im0, ax=ax[0,0])
ax[0,0].set_title('Original')

im0 = ax[1,0].imshow(sources[1][0])
fig.colorbar(im0, ax=ax[1,0])
ax[1,0].set_title('Original')

# im1 = ax[1].imshow(abs(measured_noisy[1,0,250:700,550:770]))
im1 = ax[0,1].imshow(abs(measured_noisy[0][0]))
fig.colorbar(im1, ax=ax[0,1])
ax[0,1].set_title('Measurement')

im1 = ax[1,1].imshow(abs(measured_noisy[1][0]))
fig.colorbar(im1, ax=ax[1,1])
ax[1,1].set_title('Measurement')

# im2 = ax[2].imshow(abs(recon[0,0,250:700,550:770]))
im2 = ax[0,2].imshow(abs(recon[0][0]))
fig.colorbar(im2, ax=ax[0,2])
ax[0,2].set_title('Reconstruction\n($\lambda$={:.1e}, order={:})'.format(lam,order))

# im2 = ax[3].imshow(abs(recon[1,0,250:700,550:770]))
im2 = ax[1,2].imshow(abs(recon[1][0]))
fig.colorbar(im2, ax=ax[1,2])
ax[1,2].set_title('Reconstruction\n($\lambda$={:.1e}, order={:})'.format(lam,order))
plt.show()

def figo(x):
    figo, ax = plt.subplots(nrows=2, ncols=2)
    im00 = ax[0,0].imshow(np.abs(fftshift(x[0,0,:,:])))
    figo.colorbar(im00, ax=ax[0,0])

    im01 = ax[0,1].imshow(np.abs(fftshift(x[0,1,:,:])))
    figo.colorbar(im01, ax=ax[0,1])

    im10 = ax[1,0].imshow(np.abs(fftshift(x[1,0,:,:])))
    figo.colorbar(im10, ax=ax[1,0])

    im11 = ax[1,1].imshow(np.abs(fftshift(x[1,1,:,:])))
    figo.colorbar(im11, ax=ax[1,1])
    plt.show()

figo(measurement_psf_dfts)
figo(SIG_inv)
