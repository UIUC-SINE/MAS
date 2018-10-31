#!/usr/bin/env python3
# Ulas Kamaci 2018-10-27

from mas.psf_generator import PSFs, PhotonSieve, circ_incoherent_psf
from mas.plotting import fourier_slices, plotter4d
from mas.sse_cost import block_mul, block_inv, block_herm, SIG_e_dft
from mas.forward_model import get_measurements, add_noise, size_equalizer
from mas.deconvolution import Reconstruction, tikhonov
from mas.csbs import csbs
from mas import sse_cost
from matplotlib import pyplot as plt
from matplotlib.figure import figaspect
import numpy as np
import scipy.misc

nonoise = False
num_instances = 1
aa = bb = 512
psf_width = 301
source_wavelengths = np.array([33.4e-9, 33.5e-9])
num_sources = len(source_wavelengths)

sources = np.array(
    [scipy.misc.face(gray=True)[:aa, :bb]] * num_sources
)[:, np.newaxis, :, :]
sources[1,0] = scipy.misc.face(gray=True)[-aa:, -bb:]

# [aa, bb] = np.load('/home/kamo/tmp/20181019_512c_0193.npy').shape
# sources = np.zeros((len(source_wavelengths),1,aa,bb))
# sources[0,0] = np.load('/home/kamo/tmp/20181019_512c_0193.npy')
# sources[1,0] = np.load('/home/kamo/tmp/20181019_512c_0211.npy')

lam = 1e-2
csbs_order = 1
tikhonov_order = 1
# lam = 0

ps = PhotonSieve(diameter=8e-2)
# generate psfs
psfs = PSFs(
    ps,
    # measurement_wavelengths=source_wavelengths,
    source_wavelengths=source_wavelengths,
    psf_generator=circ_incoherent_psf,
    image_width=psf_width,
    num_copies=5
)

# ----- CSBS -----

csbs(psfs, sse_cost, 10, lam=lam, order=csbs_order)

measured = get_measurements(sources=sources, psfs=psfs)

# take multiple measurements with different noise
measured_noisy_instances = add_noise(
    # duplicate measurement `num_iters` times
    np.repeat(measured[np.newaxis], num_instances, axis=0),
    snr_db=10, nonoise = nonoise
)

recon = Reconstruction(
    sources=sources,
    measurements=measured_noisy_instances,
    psfs=psfs,
    deconvolver=tikhonov,
    lam=lam,
    order=tikhonov_order
)
