#!/usr/bin/env python3
# Ulas Kamaci 2018-10-27

from mas.psf_generator import PSFs, PhotonSieve, circ_incoherent_psf
from mas.plotting import fourier_slices, plotter4d
from mas.block import block_mul, block_inv, block_herm
from mas.deconvolution.common import dctmtx
from mas.deconvolution.admm import patch_based, TV, bm3d_pnp, dncnn_pnp
from mas.forward_model import get_measurements, add_noise, size_equalizer, rectangle_adder
from mas.deconvolution import tikhonov, sparsepatch, admm, strollr, ista
from mas.csbs import csbs
from mas import sse_cost
from matplotlib import pyplot as plt
from functools import partial
from matplotlib.figure import figaspect
from skimage.metrics import structural_similarity as ssim
from scipy.io import readsav
import numpy as np
import imageio, pickle, h5py
from keras.models import load_model
from mas.data import strands, strands_ext

# model = load_model('/home/kamo/Projects/DnCNN-keras/snapshot/save_DnCNN_maxcount500_2019-04-21-16-55-59/model_10.h5')
# model = load_model('/home/kamo/Projects/DnCNN-keras/snapshot/save_DnCNN_maxcount10_2019-04-25-01-09-20/model_20.h5')

# %% meas -------------------------------------

deconvolver = admm
thresholding = 'hard' # 'hard' or 'soft' - NotImplemented
no_noise = False
meas_size = [128, 128]
# source_wavelengths = np.array([33.5e-9])
source_wavelengths = np.array([33.4e-9, 33.5e-9])
num_sources = len(source_wavelengths)

sources = strands_ext[0:num_sources]
# source1 = np.array(h5py.File('/home/kamo/Research/mas/nanoflare_videos/NanoMovie0_2000strands_94.h5')['NanoMovie0_2000strands_94'])[0]
# source2 = np.array(h5py.File('/home/kamo/Research/mas/nanoflare_videos/NanoMovie0_2000strands_94.h5')['NanoMovie0_2000strands_94'])[-1]
# source1 = source1 / source1.max()
# source2 = source2 / source2.max()
# sources[0], sources[1] = source1, source2
sources_ref = size_equalizer(sources, ref_size=meas_size)

ps = PhotonSieve(diameter=16e-2, smallest_hole_diameter=7e-6)

# generate psfs
psfs = PSFs(
    ps,
    sampling_interval=3.5e-6,
    measurement_wavelengths=source_wavelengths,
    source_wavelengths=source_wavelengths,
    psf_generator=circ_incoherent_psf,
    # image_width=psf_width,
    num_copies=1
)

measured = get_measurements(sources=sources, mode='valid', meas_size=meas_size, psfs=psfs)
measured_noisy = add_noise(measured, max_count=10, model='Poisson')

# %% recon ------------------------------------

recon = tikhonov(
    # sources=sources_ref,
    measurements=measured_noisy,
    psfs=psfs,
    tikhonov_lam=2e-1,
    tikhonov_order=1
)
if deconvolver==admm:
    recon = admm(
        sources=sources_ref,
        measurements=measured_noisy,
        psfs=psfs,
        regularizer=partial(
            # patch_based
            # TV
            bm3d_pnp
            # dncnn_pnp, model=model
            ),
        recon_init=recon,
        iternum=20,
        periter=5,
        nu=10**-1.2,
        lam=[10**-2.5]*num_sources,
    )
elif deconvolver==strollr:
    recon = strollr(
        sources=sources_ref,
        measurements=measured_noisy,
        psfs=psfs,
        recon_init=recon,
        iternum=12,
        periter=1,
        lr=5e-3,
        theta=[1,1],
        s=0,
        lam=2e-1,
        patch_shape=(6,6,1),
        transform=dctmtx((6,6,8)),
        learning=True,
        window_size=(30,30),
        group_size=70,
        group_size_s=8
    )
elif deconvolver==sparsepatch:
    recon = sparsepatch(
        sources=sources_ref,
        measurements=measured_noisy,
        psfs=psfs,
        recon_init=recon,
        iternum=500,
        periter=50,
        nu=1e-2,
        lam=1e-5,
        patch_shape=(6,6,1),
        transform=dctmtx((6,6,1)),
        learning=True
    )
elif deconvolver==ista:
    recon = ista(
        sources=sources_ref,
        measurements=measured_noisy,
        psfs=psfs,
        iterations=60,
        lam=10**-6,
        time_step=10**-2,
        liveplot=True
    )

###### COMPUTE PERFORMANCE METRICS ######
ssim_ = np.zeros(num_sources)
mse_ = np.mean((sources_ref - recon)**2, axis=(1, 2))
psnr_ = 20 * np.log10(np.max(sources_ref, axis=(1,2))/np.sqrt(mse_))
for i in range(num_sources):
    ssim_[i] = ssim(sources_ref[i], recon[i],
        data_range=np.max(recon[i])-np.min(recon[i]))

plotter4d(recon,
    cmap='gist_heat',
    figsize=(5.6,8),
    title='Recon. SSIM={}\n Recon. PSNR={}'.format(ssim_, psnr_)
)
print('ssim:{:.3f}, psnr:{:.2f}'.format(np.mean(ssim_, axis=0),np.mean(psnr_, axis=0)))

plotter4d(measured_noisy,
    cmap='gist_heat',
    figsize=(5.6,8),
    title='Noisy Meas'
)
# plotter4d(sources,
#     title='Orig',
#     figsize=(5.6,8),
#     cmap='gist_heat'
# )
