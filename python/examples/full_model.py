# %% import_load
#!/usr/bin/env python3
# Ulas Kamaci 2018-10-27

from mas.psf_generator import PSFs, PhotonSieve, circ_incoherent_psf
from mas.plotting import fourier_slices, plotter4d
from mas.sse_cost import block_mul, block_inv, block_herm, SIG_e_dft
from mas.forward_model import get_measurements, add_noise, size_equalizer, rectangle_adder
from mas.deconvolution import tikhonov, sparsepatch, admm, dctmtx, strollr
from mas.csbs import csbs
from mas import sse_cost
from matplotlib import pyplot as plt
from matplotlib.figure import figaspect
from skimage.measure import compare_ssim as ssim
from scipy.io import readsav
import numpy as np
import imageio, pickle, h5py
from keras.models import load_model

# model = load_model('/home/kamo/Projects/DnCNN-keras/snapshot/save_DnCNN_maxcount500_2019-04-21-16-55-59/model_10.h5')
# model = load_model('/home/kamo/Projects/DnCNN-keras/snapshot/save_DnCNN_maxcount10_2019-04-25-01-09-20/model_20.h5')

# %% meas
deconvolver = admm
regularizer = 'bm3d_pnp' # ['patch_based', 'TV', 'strollr', 'bm3d_pnp', 'dncnn']
thresholding = 'hard' # 'hard' or 'soft' - NotImplemented
nonoise = False
num_instances = 1
psf_width = 201
source_wavelengths = np.array([33.5e-9])
# source_wavelengths = np.array([33.4e-9, 33.5e-9])
num_sources = len(source_wavelengths)

source1 = size_equalizer(np.array(h5py.File('/home/kamo/Research/mas/nanoflare_videos/NanoMovie0_2000strands_94.h5')['NanoMovie0_2000strands_94'])[0], (160,160))
# source2 = size_equalizer(np.array(h5py.File('/home/kamo/Research/mas/nanoflare_videos/NanoMovie0_2000strands_94.h5')['NanoMovie0_2000strands_94'])[1999], (160,160))
# source1 = rectangle_adder(image=np.zeros((100,100)), size=(30,30), upperleft=(35,10))
# source2 = 10 * rectangle_adder(image=np.zeros((100,100)), size=(30,30), upperleft=(35,60))
# source1 = readsav('/home/kamo/Research/mas/nanoflare_videos/old/movie0_1250strands_335.sav',python_dict=True)['movie'][500]
# source2 = readsav('/home/kamo/Research/mas/nanoflare_videos/old/movie0_1250strands_94.sav',python_dict=True)['movie'][500]
[aa, bb] = source1.shape
meas_size = tuple(np.array([aa,bb]) - 0)
sources = np.zeros((len(source_wavelengths),1,aa,bb))
sources[0,0] = source1 / source1.max()
# sources[1,0] = source2 / source2.max()
# sources = sources / sources.max()

ps = PhotonSieve(diameter=6e-2, smallest_hole_diameter=20e-6)
# generate psfs
psfs = PSFs(
    ps,
    sampling_interval=10e-6,
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
        measured_noisy_instances[i, j, 0] = add_noise(
            measured[j,0],
            snr=10, maxcount=100, nonoise=nonoise, model='Poisson'
            # snr=100, nonoise=nonoise, model='Gaussian'
        )
if len(measured_noisy_instances.shape) == 4:
    measured_noisy_instances = measured_noisy_instances[np.newaxis]

plotter4d(measured_noisy_instances[0],
    cmap='gist_heat',
    figsize=(5.6,8),
    title='Noisy Meas'
)
# plotter4d(sources,
#     title='Orig',
#     figsize=(5.6,8),
#     cmap='gist_heat'
# )


# %% recon
if deconvolver==tikhonov:
    recon = tikhonov(
        sources=sources,
        measurements=np.fft.fftshift(measured_noisy_instances, axes=(3,4))[0],
        psfs=psfs,
        tikhonov_lam=5e-2,
        tikhonov_order=1
    )
elif deconvolver==admm:
    recon = admm(
        sources=sources,
        measurements=np.fft.fftshift(measured_noisy_instances, axes=(3,4))[0],
        psfs=psfs,
        regularizer=regularizer,
        recon_init_method='tikhonov',
        iternum=30,
        nu=14e-2,
        lam=5e-5,
        tikhonov_lam=1e-1,
        tikhonov_order=1,
        patch_shape=(6,6,1),
        transform=dctmtx((6,6,8)),
        learning=True,
        window_size=(30,30),
        group_size=70
        # model=model
    )
elif deconvolver==strollr:
    recon = strollr(
        sources=sources,
        measurements=np.fft.fftshift(measured_noisy_instances, axes=(3,4))[0],
        psfs=psfs,
        recon_init_method='tikhonov',
        tikhonov_lam=1e-1,
        tikhonov_order=1,
        iternum=12,
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
        sources=sources,
        measurements=np.fft.fftshift(measured_noisy_instances, axes=(3,4))[0],
        psfs=psfs,
        recon_init_method='tikhonov',
        tikhonov_lam=2e-2,
        tikhonov_order=1,
        iternum=500,
        nu=1e-2,
        lam=1e-5,
        patch_shape=(6,6,1),
        transform=dctmtx((6,6,1)),
        learning=True
    )

###### COMPUTE PERFORMANCE METRICS ######
ssim_ = np.zeros(num_sources)
mse_ = np.mean((sources - recon)**2, axis=(1, 2, 3))
psnr_ = 20 * np.log10(np.max(sources, axis=(1,2,3))/np.sqrt(mse_))
for i in range(num_sources):
    ssim_[i] = ssim(sources[i,0], recon[i,0],
        data_range=np.max(recon[i,0])-np.min(recon[i,0]))

plotter4d(recon,
    cmap='gist_heat',
    figsize=(5.6,8),
    title='Recon. SSIM={}\n Recon. PSNR={}'.format(ssim_, psnr_)
)
print('ssim:{:.3f}, psnr:{:.2f}'.format(ssim_[0],psnr_[0]))
