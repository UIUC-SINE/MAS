# %% init

import numpy as np
import matplotlib.pyplot as plt
from mas.psf_generator import PSFs, PhotonSieve, circ_incoherent_psf
from mas import sse_cost
from mas.csbs import csbs
from mas.forward_model import get_measurements, add_noise
from mas.deconvolution import tikhonov
from mas.measure import compare_ssim, compare_psnr
from mas.plotting import plotter4d

# source_wavelengths = np.array((90.359e-9, 90.399e-9))
# diameter = 5e-2
# smallest_hole_diameter = 15e-6
# csbs_lam = 5e-5
source_wavelengths = np.array((33.4e-9, 33.402e-9))
diameter = 16e-2
smallest_hole_diameter = 7e-6
print('DOF_separation = {}'.format(
    diameter * (source_wavelengths[1] - source_wavelengths[0]) /
    (2 * smallest_hole_diameter * source_wavelengths[0])
    ))
sources = np.load('sources.npy')
order = 1
csbs_lam = 5e-4

ps = PhotonSieve(diameter=diameter, smallest_hole_diameter=smallest_hole_diameter)
psfs_csbs = PSFs(
    ps,
    sampling_interval=3.5e-6,
    measurement_wavelengths=30,
    source_wavelengths=source_wavelengths,
    psf_generator=circ_incoherent_psf,
    image_width=501,
    # cropped_width=51,
    num_copies=10
)

psfs_focus = PSFs(
    ps,
    sampling_interval=3.5e-6,
    measurement_wavelengths=source_wavelengths,
    source_wavelengths=source_wavelengths,
    psf_generator=circ_incoherent_psf,
    image_width=501,
    # cropped_width=psfs_csbs.psfs.shape[-1],
    num_copies=6
)

csbs(psfs_csbs, sse_cost, 12, lam=csbs_lam, order=order)

# %% measure
psnr_focus, psnr_csbs, ssim_focus, ssim_csbs = [], [], [], []
measured_focus = get_measurements(sources=sources, psfs=psfs_focus)
measured_csbs = get_measurements(sources=sources, psfs=psfs_csbs)
for i in range(50):
    measured_noisy_focus = add_noise(measured_focus, dbsnr=15, model='Gaussian')
    measured_noisy_csbs = add_noise(measured_csbs, dbsnr=15, model='Gaussian')

    recon_focus = tikhonov(psfs=psfs_focus, measurements=measured_noisy_focus, tikhonov_lam=1e-3, tikhonov_order=order)
    recon_csbs = tikhonov(psfs=psfs_csbs, measurements=measured_noisy_csbs, tikhonov_lam=1e-3, tikhonov_order=order)

    psnr_focus.append(compare_psnr(sources, recon_focus))
    psnr_csbs.append(compare_psnr(sources, recon_csbs))
    ssim_focus.append(np.sum(compare_ssim(sources, recon_focus)))
    ssim_csbs.append(np.sum(compare_ssim(sources, recon_csbs)))

print('psnr_focus:{} \npsnr_csbs:{} \nssim_focus:{} \nssim_csbs:{}'.format(
    np.array(psnr_focus).mean(), np.array(psnr_csbs).mean(), np.array(ssim_focus).mean(), np.array(ssim_csbs).mean()
))

plotter4d(recon_focus, title='recon_focus \n ssim:{}  psnr:{}'.format(np.array(ssim_focus).mean(), np.array(psnr_focus).mean()), colorbar=True)
plotter4d(recon_csbs, title='recon_csbs \n ssim:{}  psnr:{}'.format(np.array(ssim_csbs).mean(), np.array(psnr_csbs).mean()), colorbar=True)
plotter4d(sources, title='sources')

# %% plotting

z = np.zeros((160, 160, 3))
z[:, :, 0] = sources[0]
z[:, :, 1] = sources[1]
# plt.imsave('figures/sources.png', z)

z = np.zeros((160, 160, 3))
z[:, :, 0] = measured_noisy_focus[5]
z[:, :, 1] = measured_noisy_focus[5]
z[:, :, 2] = measured_noisy_focus[5]
# plt.imsave('figures/measured1.png', z)
z = np.zeros((160, 160, 3))
z[:, :, 0] = measured_noisy_csbs[6]
z[:, :, 1] = measured_noisy_csbs[6]
z[:, :, 2] = measured_noisy_csbs[6]
# plt.imsave('figures/measured2.png', z)

z = np.zeros((160, 160, 3))
z[:, :, 0] = recon_focus[0]
# plt.imsave('figures/recon_focus1.png', z)
z = np.zeros((160, 160, 3))
z[:, :, 1] = recon_focus[1]
# plt.imsave('figures/recon_focus2.png', z)

z = np.zeros((160, 160, 3))
z[:, :, 0] = recon_csbs[0]
# plt.imsave('figures/recon_csbs1.png', z)

z = np.zeros((160, 160, 3))
z[:, :, 1] = recon_csbs[1]
# plt.imsave('figures/recon_csbs2.png', z)

plt.figure(figsize=(4, 3))
plt.stem(psfs_csbs.copies, 'ko-', basefmt=' ')
plt.axvline(15, color='#cc0000', linewidth=3.5)
plt.axvline(16, color='#73d216', linewidth=3.5)
plt.axis('off')
plt.savefig('figures/csbs_copies.png')
