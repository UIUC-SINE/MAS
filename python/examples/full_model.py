#!/usr/bin/env python3
# Ulas Kamaci 2018-10-27

from mas.psf_generator import PSFs, PhotonSieve, circ_incoherent_psf
from mas.plotting import fourier_slices, plotter4d
from mas.sse_cost import block_mul, block_inv, block_herm, SIG_e_dft
from mas.forward_model import get_measurements, add_noise, size_equalizer, rectangle_adder
from mas.deconvolution import Reconstruction, tikhonov, sparsepatch, admm, admm_tv, dctmtx, strollr, fista_tv, fista_dct
from mas.csbs import csbs
from mas import sse_cost
from matplotlib import pyplot as plt
from matplotlib.figure import figaspect
from scipy.io import readsav
import numpy as np
import imageio, pickle, h5py

deconvolver = tikhonov
sparsifyer = 'patch_based' # 'patch_based' or 'TV'
thresholding = 'hard' # 'hard' or 'soft'
learning = True
nonoise = False
lcurve = False
store_recons = True
num_instances = 1
# aa = bb = 512
psf_width = 201
# source_wavelengths = np.array([9.4e-9])
source_wavelengths = np.array([33.4e-9, 33.5e-9])
num_sources = len(source_wavelengths)

# sources = np.array(
#     [scipy.misc.face(gray=True)[:aa, :bb]] * num_sources
# )[:, np.newaxis, :, :]
# sources[1,0] = scipy.misc.face(gray=True)[-aa:, -bb:]

# source1 = np.array(h5py.File('/home/kamo/Research/mas/nanoflare_videos/NanoMovie0_2000strands_94.h5')['NanoMovie0_2000strands_94'])[0]
source1 = rectangle_adder(image=np.zeros((100,100)), size=(30,30), upperleft=(35,10))
source2 = 10 * rectangle_adder(image=np.zeros((100,100)), size=(30,30), upperleft=(35,60))
# source2 = 10 * source1
# source1 = np.load('/home/kamo/imo.npy')
# source = imageio.imread('/home/kamo/tmp/cameraman.png')[:,:,1]
# source1 = np.load('/home/kamo/tmp/20181019_512c_0193.npy')[123:123+205,283:283+256]
# source2 = np.load('/home/kamo/tmp/20181019_512c_0211.npy')[123:123+205,283:283+256]
# source1 = np.load('/home/kamo/tmp/20181019_512c_0193.npy')[23:123+305,183:283+356]
# source2 = np.load('/home/kamo/tmp/20181019_512c_0211.npy')[23:123+305,183:283+356]
# source1 = readsav('/home/kamo/Research/mas/nanoflare_videos/old/movie0_1250strands_335.sav',python_dict=True)['movie'][500]
# source2 = readsav('/home/kamo/Research/mas/nanoflare_videos/old/movie0_1250strands_94.sav',python_dict=True)['movie'][500]
# source1 = source1/np.linalg.norm(source1)
# source2 = source2/np.linalg.norm(source2)
# source2 = source2 - np.mean(source2)
# source2 = source2 - np.mean(source2)
# source1 = size_equalizer(source1, (300,300))
[aa, bb] = source1.shape
meas_size = tuple(np.array([aa,bb]) - 0)
sources = np.zeros((len(source_wavelengths),1,aa,bb))
sources[0,0] = source1 / source1.max()
sources[1,0] = source2 / source2.max()
# sources[1,0] = (sources[1,0] + 1) / 2
# sources = sources / sources.max()

csbs_order = 1
tikhonov_order = 1
tikhonov_lam = 8e-3
tikhonov_matrix = 'derivative' # 'covariance' or 'derivative'
tikhonov_scale = 'full' # 'patch' or 'full'
iterproduct = True
cmap = 'gist_heat'

lam = 1e-3
nu = 1e-1
s = 0
lr = 2e-3
M = 70
l = 8
theta = (2e-1, 1e20)
sparsity_ratio = 0.2
sparsity_threshold = np.sqrt(2*np.array(lam)/np.array(nu))
tv = 'aniso'
recon_init_method = 'tikhonov'
patch_shape = (6,6,1)
window_size = (30,30)
transform = dctmtx((patch_shape[0],patch_shape[1],l))
#FIXME
# transform = dctmtx((patch_shape[0],patch_shape[1],1))
maxiter = 4
lcurve_param = {'tikhonov_lam': tikhonov_lam}

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
# ----- CSBS -----
# csbs(psfs, sse_cost, 2, lam=tikhonov_lam, order=csbs_order)

# pickle_in = open('/home/kamo/tmp/psfs10.pickle', 'rb')
# psfs = pickle.load(pickle_in)
# psfs.copies=np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 10., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 10., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
measured = get_measurements(real=True, sources=sources, psfs=psfs, meas_size=meas_size, mode='circular')
sources = size_equalizer(sources, meas_size)
# take multiple measurements with different noise
measured_noisy_instances = np.zeros((num_instances,)+measured.shape)
for i in range(num_instances):
    for j in range(measured.shape[0]):
        measured_noisy_instances[i, j, 0] = add_noise(
            measured[j,0],
            snr=10, maxcount=500, nonoise=nonoise, model='Poisson'
            # snr=100, nonoise=nonoise, model='Gaussian'
        )
if len(measured_noisy_instances.shape) == 4:
    measured_noisy_instances = measured_noisy_instances[np.newaxis]

# measured_noisy_instances = np.load('/tmp/ms.npy')
# plotter4d(np.fft.fftshift(measured_noisy_instances[0], axes=(2,3)),
#     title='Noisy Meas',
#     cmap='gist_heat'
# )
plotter4d(sources,
    fignum=1,
    title='Orig',
    cmap='gist_heat'
)

recon = Reconstruction(
    sources=sources,
    measurements=np.fft.fftshift(measured_noisy_instances, axes=(3,4)),
    psfs=psfs,
    deconvolver=deconvolver,
    recon_init_method=recon_init_method,
    iterproduct=iterproduct,
    tikhonov_lam=tikhonov_lam,
    tikhonov_order=tikhonov_order,
    tikhonov_scale=tikhonov_scale,
    tikhonov_matrix=tikhonov_matrix,
    patch_shape=patch_shape,
    transform=transform,
    learning=learning,
    tv=tv,
    nu=nu,
    lam=lam,
    sparsity_ratio=sparsity_ratio,
    sparsity_threshold=sparsity_threshold,
    maxiter=maxiter,
    lcurve=lcurve,
    lcurve_param=lcurve_param,
    store_recons=store_recons,
    thresholding=thresholding,
    sparsifyer=sparsifyer,
    s=s,
    lr=lr,
    M=M,
    theta=theta,
    l=l,
    window_size=window_size
)

plotter4d(recon.reconstructed,
    title='Recon. SSIM={}\n Recon. PSNR={}'.format(recon.ssim[-1,-1], recon.psnr[-1,-1]),
    cmap=cmap)
