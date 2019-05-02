#!/usr/bin/env python3
# Ulas Kamaci 2019-04-19

from mas.psf_generator import PSFs, PhotonSieve, circ_incoherent_psf
from mas.plotting import fourier_slices, plotter4d
from mas.sse_cost import block_mul, block_inv, block_herm, SIG_e_dft, get_LAM
from mas.forward_model import get_measurements, add_noise, size_equalizer, rectangle_adder
from mas.deconvolution import tikhonov, sparsepatch, admm, dctmtx, strollr
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim
from scipy.io import readsav
import numpy as np
import imageio, pickle, h5py

num_instances = 1
tikhonov_order = 1
tikhonov_lam = 1e-2
if type(tikhonov_lam) is np.int or type(tikhonov_lam) is np.float:
    tikhonov_lam = [tikhonov_lam]
psf_width = 201
# source_wavelengths = np.array([9.4e-9])
source_wavelengths = np.array([33.4e-9, 33.5e-9])
num_sources = len(source_wavelengths)

source1 = size_equalizer(np.array(h5py.File('/home/kamo/Research/mas/nanoflare_videos/NanoMovie0_2000strands_94.h5')['NanoMovie0_2000strands_94'])[0], (100,100))
source2 = size_equalizer(np.array(h5py.File('/home/kamo/Research/mas/nanoflare_videos/NanoMovie0_2000strands_94.h5')['NanoMovie0_2000strands_94'])[250], (100,100))
# source1 = rectangle_adder(image=np.zeros((100,100)), size=(30,30), upperleft=(35,10))
# source2 = 10 * rectangle_adder(image=np.zeros((100,100)), size=(30,30), upperleft=(35,60))
# source1 = readsav('/home/kamo/Research/mas/nanoflare_videos/old/movie0_1250strands_335.sav',python_dict=True)['movie'][500]
# source2 = readsav('/home/kamo/Research/mas/nanoflare_videos/old/movie0_1250strands_94.sav',python_dict=True)['movie'][500]
[aa, bb] = source1.shape
meas_size = tuple(np.array([aa,bb]) - 0)
sources = np.zeros((len(source_wavelengths),1,aa,bb))
sources[0,0] = source1 / source1.max()
sources[1,0] = source2 / source2.max()
# sources = sources / sources.max()

cmap = 'gist_heat'

ps = PhotonSieve(diameter=10e-2)
# generate psfs
psfs = PSFs(
    ps,
    sampling_interval=3.5e-6,
    measurement_wavelengths=source_wavelengths,
    source_wavelengths=source_wavelengths,
    psf_generator=circ_incoherent_psf,
    image_width=psf_width,
    num_copies=1
)

measured = get_measurements(real=True, sources=sources, psfs=psfs, meas_size=meas_size, mode='circular')
sources = size_equalizer(sources, meas_size)
# take multiple measurements with different noise
measured_noisy_instances = np.zeros((num_instances,)+measured.shape)
for i in range(num_instances):
    for j in range(measured.shape[0]):
        measured_noisy_instances[i, j, 0] = np.fft.fftshift(add_noise(
            measured[j,0],
            snr=10, max_count=500, model='Poisson'
            # snr=100, no_noise=no_noise, model='Gaussian'
        ))
if len(measured_noisy_instances.shape) == 4:
    measured_noisy_instances = measured_noisy_instances[np.newaxis]

plotter4d(sources,
    title='Orig',
    figsize=(5.6,8),
    cmap='gist_heat'
)

[k,num_sources,aa,bb] = psfs.selected_psfs.shape[:2] + sources.shape[2:]
LAM = get_LAM(rows=aa,cols=bb,order=tikhonov_order)
ssim = np.zeros((len(tikhonov_lam), num_instances, num_sources))
psnr = np.zeros((len(tikhonov_lam), num_instances, num_sources))
for i in range(len(tikhonov_lam)):
    SIG_inv = block_inv(
        psfs.selected_GAM +
        tikhonov_lam[i] * np.einsum('ij,kl', np.eye(num_sources), LAM)
    )
    for j in range(num_instances):
        # DFT of the kernel corresponding to (D^TD)
        recon = np.real(
            np.fft.ifft2(
                    block_mul(
                        SIG_inv,
                        block_mul(
                            psfs.selected_psf_dfts_h,
                            np.fft.fft2(measured_noisy_instances[j])
                        )
                    )
            )
        )
        ###### COMPUTE PERFORMANCE METRICS ######
        mse = np.mean((sources - recon)**2, axis=(1, 2, 3))
        psnr[i,j] = 20 * np.log10(np.max(sources, axis=(1,2,3))/np.sqrt(mse))
        for p in range(num_sources):
            ssim[i,j,p] = compare_ssim(sources[p,0], recon[p,0],
                data_range=np.max(recon[p,0])-np.min(recon[p,0]))

plotter4d(recon,
    cmap='gist_heat',
    figsize=(5.6,8),
    title='Recon. SSIM={}\n Recon. PSNR={}'.format(ssim[-1,-1], psnr[-1,-1])
)
