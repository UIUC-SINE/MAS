#!/bin/env python3
# Evan Widloski, Ulas Kamaci - 2019-10-18

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mas.psf_generator import PSFs, PhotonSieve, circ_incoherent_psf
from mas import sse_cost
from mas.csbs import csbs
from mas.forward_model import (
    get_measurements, add_noise, dof2wavelength, wavelength2dof
)
from mas.deconvolution import tikhonov
from mas.measure import compare_ssim, compare_psnr
from mas.plotting import plotter4d
from mas.misc import combination_experiment
from cachalot import Cache
from mas.data import strands
from mas.strand_generator import strands as strand_gen
import seaborn as sns; sns.set()
from tqdm import tqdm
from pathlib import Path
import os
from copy import copy

cache_path = Path('/dev/shm/icip.json')
cache_path2 = Path('/dev/shm/icip2.json')
if cache_path.exists():
    os.remove(cache_path)
if cache_path2.exists():
    os.remove(cache_path2)

strands = np.array([strand_gen(image_width=75) for _ in range(4)])

order = 1
base_wavelength = 33.4e-9
diameter = 0.1
ps = PhotonSieve(diameter=diameter)
# ps = PhotonSieve(diameter=diameter, smallest_hole_diameter=smallest_hole_diameter)

# smallest_hole_diameter = 7e-6
def lam_lookup(df, num_sources, snr):
    # put rows into groups indexed by these vars
    df = df.groupby(['num_sources', 'snr', 'tik_lam']).mean().reset_index()
    # get the group we are interested
    df = df.query('snr == @snr and num_sources == @num_sources')
    # get lambda corresponding to max ssim_focus
    return float(df[df.ssim_focus == df.ssim_focus.max()].tik_lam)

@Cache(path=cache_path, size=2)
def get_psfs(num_sources, separation, csbs_lam=None, grid_width=None,
             measurement_wavelengths=None, no_dc=None, autocrop=False, include_csbs=True,
             sampling_interval=3.5e-6, image_width=301
):

    source_wavelengths = dof2wavelength(
        base_wavelength=base_wavelength,
        dof=separation*np.arange(num_sources),
        ps=ps
    )

    psfs_focus = PSFs(
        ps,
        sampling_interval=sampling_interval,
        measurement_wavelengths=source_wavelengths,
        source_wavelengths=source_wavelengths,
        psf_generator=circ_incoherent_psf,
        image_width=301,
        cropped_width=image_width if not autocrop else None,
        num_copies=6,
        grid_width=grid_width,
    )

    if include_csbs:
        psfs_csbs = PSFs(
            ps,
            sampling_interval=sampling_interval,
            measurement_wavelengths=measurement_wavelengths,
            source_wavelengths=source_wavelengths,
            psf_generator=circ_incoherent_psf,
            image_width=301,
            cropped_width=image_width if not autocrop else None,
            num_copies=6,
            grid_width=grid_width,
        )

        psfs_csbs = csbs(
            psfs_csbs,
            sse_cost,
            6 * num_sources,
            lam=csbs_lam,
            order=order,
            no_dc=no_dc
        )

        return source_wavelengths, psfs_focus, psfs_csbs
    else:
        return source_wavelengths, psfs_focus, None


# @Cache(path=cache_path2, size=1)
# def get_strands(num_sources, image_width):
#     strands = [strand_gen(image_width=image_width) for _ in range(num_sources)]
#     return np.array(strands)


def experiment(*, num_sources, separation, snr, csbs_lam, grid_width,
               measurement_wavelengths, no_dc, image_width, autocrop, all=False, **kwargs):

    # sources = get_strands(num_sources, image_width)
    sources = copy(strands[:num_sources])

    if csbs_lam is None:
        params_df = pd.read_pickle('result_params.pkl')
        csbs_lam = lam_lookup(params_df, num_sources, snr)
    source_wavelengths, psfs_focus, psfs_csbs = get_psfs(
        num_sources=num_sources,
        separation=separation,
        csbs_lam=csbs_lam,
        grid_width=grid_width,
        measurement_wavelengths=measurement_wavelengths,
        no_dc=no_dc,
        image_width=image_width,
        autocrop=autocrop
    )

    measured_focus = get_measurements(sources=sources, psfs=psfs_focus)
    measured_csbs = get_measurements(sources=sources, psfs=psfs_csbs)

    measured_noisy_focus = add_noise(measured_focus, dbsnr=snr, model='Gaussian')
    measured_noisy_csbs = add_noise(measured_csbs, dbsnr=snr, model='Gaussian')

    recon_focus = tikhonov(
        psfs=psfs_focus,
        measurements=measured_noisy_focus,
        tikhonov_lam=csbs_lam,
        tikhonov_order=order
    )
    recon_csbs = tikhonov(
        psfs=psfs_csbs,
        measurements=measured_noisy_csbs,
        tikhonov_lam=csbs_lam,
        tikhonov_order=order
    )

    if no_dc:
        sources -= sources.mean(axis=(1,2))[:, np.newaxis, np.newaxis]
        recon_focus -= recon_focus.mean(axis=(1,2))[:, np.newaxis, np.newaxis]
        recon_csbs -= recon_csbs.mean(axis=(1,2))[:, np.newaxis, np.newaxis]

    ssim_focus = np.sum(compare_ssim(truth=sources, estimate=recon_focus))
    ssim_csbs = np.sum(compare_ssim(truth=sources, estimate=recon_csbs))
    psnr_focus = np.sum(compare_psnr(truth=sources, estimate=recon_focus))
    psnr_csbs = np.sum(compare_psnr(truth=sources, estimate=recon_csbs))

    ratio = ssim_csbs / ssim_focus

    return {
        **{
        'ratio': ratio,
        'ssim_focus': ssim_focus,
        'ssim_csbs': ssim_csbs,
        'psnr_focus': psnr_focus,
        'psnr_csbs': psnr_csbs,
        },
        **(
            {
                'psfs_csbs': psfs_csbs,
                'psfs_focus': psfs_focus,
                'source_wavelengths': source_wavelengths,
                'measured_focus': measured_focus,
                'measured_csbs': measured_csbs,
                'measured_noisy_focus': measured_noisy_focus,
                'measured_noisy_csbs': measured_noisy_csbs,
                'recon_focus': recon_focus,
                'recon_csbs': recon_csbs,
                'num_copies': psfs_csbs.num_copies,
                'copies': psfs_csbs.copies,
                'copies_history': psfs_csbs.copies_history
            } if all else {}
        )
    }


def experiment_param(*, num_sources, separation, snr, tik_lam,
                     no_dc, image_width, autocrop, **kwargs):

    # sources = get_strands(num_sources, image_width)
    sources = copy(strands[:num_sources])
    source_wavelengths, psfs_focus, _ = get_psfs(
        num_sources=num_sources,
        separation=separation,
        include_csbs=False,
        image_width=image_width,
        autocrop=autocrop,
    )

    measured_focus = get_measurements(sources=sources, psfs=psfs_focus)
    measured_noisy_focus = add_noise(measured_focus, dbsnr=snr, model='Gaussian')
    recon_focus = tikhonov(
        psfs=psfs_focus,
        measurements=measured_noisy_focus,
        tikhonov_lam=tik_lam,
        tikhonov_order=order
    )
    if no_dc:
        sources -= sources.mean(axis=(1,2))[:, np.newaxis, np.newaxis]
        recon_focus -= recon_focus.mean(axis=(1,2))[:, np.newaxis, np.newaxis]

    ssim_focus = np.sum(compare_ssim(truth=sources, estimate=recon_focus))

    return {
        'ssim_focus': ssim_focus,
    }

# %% separation_experiment

separation = np.logspace(-1, 1, 15)

result = combination_experiment(
    experiment,
    num_sources=[2, 3, 4],
    separation=separation,
    snr=[5, 10, 15],
    csbs_lam=[None],
    grid_width=[5],
    measurement_wavelengths=[30],
    no_dc=[True],
    image_width=[75],
    autocrop=[False],
    repetitions=np.arange(5)
)

result.to_pickle('result.pkl')

# %% separation_plot

result_plot = result.melt(
    id_vars=['ratio', 'num_sources', 'no_dc',
             'separation', 'snr', 'csbs_lam', 'repetitions'],
    value_vars=['ssim_focus', 'ssim_csbs'],
    value_name='ssim',
    var_name='method'
)

# result_plot = result_plot.query('no_dc == True')
result_plot = result_plot.rename(columns={'num_sources': 'S'})

result_plot.ssim /= result_plot.S

plt.close()
grid = sns.FacetGrid(result_plot, col='snr', row='S', margin_titles=True)
grid.map(sns.lineplot, 'separation', 'ssim', 'method', style='no_dc', data=result_plot)
# grid.map(sns.lineplot, 'separation', 'ssim', 'method', data=result_plot)
grid.set(xscale='log')
# grid.add_legend()
# grid.set(ylim=[-0.1, 0.6])

# %% param_experiment

result = combination_experiment(
    experiment_param,
    num_sources=[2, 3, 4],
    separation=[1],
    snr=[5, 10, 15],
    tik_lam=np.logspace(-5, 0, 20),
    no_dc=[True],
    # repetitions=np.arange(50),
    repetitions=np.arange(10),
    image_width=[75],
    autocrop=[False]
)
result.to_pickle('result_params.pkl')

# %% param_plot

# x2 = pd.read_pickle('result_params2.pkl')
# x3 = pd.read_pickle('result_params3.pkl')
# x4 = pd.read_pickle('result_params4.pkl')
#
# x = pd.concat([x2, x3, x4])

# grid = sns.lineplot(x='tik_lam', y='ssim_focus', hue='num_sources', data=x)
# grid.set(xscale='log')
# grid.set_ylim([0.75, 2])

result_plot = result.melt(
    id_vars=['num_sources', 'no_dc',
             'separation', 'snr', 'tik_lam', 'repetitions'],
    value_vars=['ssim_focus'],
    value_name='ssim',
    var_name='method'
)
result_plot = result_plot.rename(columns={'num_sources': 'S'})

plt.close()
grid = sns.FacetGrid(result_plot, col='snr', row='S', margin_titles=True)
grid.map(sns.lineplot, 'tik_lam', 'ssim', 'method', style='no_dc', data=result_plot)
# grid.map(sns.lineplot, 'separation', 'ssim', 'method', data=result_plot)
grid.set(xscale='log')
grid.add_legend()
