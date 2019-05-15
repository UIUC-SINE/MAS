#!/usr/bin/env python3
# Ulas Kamaci, Evan Widloski - 2018-08-27

import numpy as np
from mas.sse_cost import block_mul, block_herm
from mas.decorators import vectorize
from scipy.stats import poisson
from scipy.signal import fftconvolve
from PIL import Image

def image_numpyer(*, inpath, outpath, size, upperleft):
    """
    Read png/jpg image, crop it with the specified size and location, and save
    it as an npy file

    Args:
        inpath (string): Path to the input png/jpg file
        outpath (string): Path to the output npy file
        size (tuple): Integer or tuple of 2 integers indicating the height and width
        of the cropped image, respectively.
        upperleft (tuple): Tuple of 2 integers indicating the coordinate of
        the upper left corner of the cropped image
    """
    if type(size) is int:
        size = (size,size)
    x = np.array(Image.open(inpath).convert('L'))
    x = x[upperleft[0]:upperleft[0]+size[0], upperleft[1]:upperleft[1]+size[1]]
    np.save(outpath, x)

def image_selector(*, inpath, size, upperleft):
    """
    Read png/jpg image, crop it with the specified coordinates and size, and
    show the full image vs cropped image.
    Puts a box around the cropped part for visualization. Box code might need
    adjustment for non-SDO images.

    Args:
        inpath (string): Path to the input png/jpg file
        size (tuple): Integer or tuple of 2 integers indicating the height and
        width of the cropped image, respectively.
        upperleft (tuple): Tuple of 2 integers indicating the coordinate of
        the upper left corner of the cropped image
    """
    if type(size) is int:
        size = (size,size)
    x = np.array(Image.open(inpath).convert('L'))
    x_c = x[upperleft[0]:upperleft[0]+size[0], upperleft[1]:upperleft[1]+size[1]]
    xmax = np.max(x[: int(x.shape[0]*0.94), :]) # to preserve the dynamic range

    # put a frame around the selected part
    t = 10
    x[upperleft[0] - t:upperleft[0],
    upperleft[1]: upperleft[1] + size[1]] = xmax

    x[upperleft[0] + size[0]+1: upperleft[0] + size[0] + t,
    upperleft[1]: upperleft[1] + size[1]] = xmax

    x[upperleft[0] - t: upperleft[0] + size[0] + t,
    upperleft[1] - t: upperleft[1]] = xmax

    x[upperleft[0] - t: upperleft[0] + size[0] + t,
    upperleft[1] + size[1] + 1: upperleft[1] + size[1] + t] = xmax

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(x[: int(x.shape[0]*0.94), :]) # crop the text to preserve dynamic range (for SDO)
    ax2.imshow(x_c)


def rectangle_adder(*, image, size, upperleft):
    """
    Add a rectangle of value one to the specified position of a given image.

    Args:
        image (ndaray): input image on which the rectangle will be added
        size (tuple): Integer or tuple of 2 integers indicating the height and
        width of the rectangle, respectively.
        upperleft (tuple): Tuple of 2 integers indicating the coordinate of
        the upper left corner of the position of the added rectangle
    """
    if type(size) is int:
        size = (size,size)
    image[
        upperleft[0]:upperleft[0]+size[0], upperleft[1]:upperleft[1]+size[1]
    ] += np.ones(size)
    return image

def get_measurements(real=False, *, sources, psfs, meas_size=None):
    """
    Convolve the sources and psfs to obtain measurements.
    Args:
        sources (ndarray): 4d array of sources
        psfs (PSFs): PSFs object containing psfs and other csbs state data
        mode (string): {'circular', 'linear'} (default='circular')
        real (bool): (default=False) whether returned measurement should be real
            type of the convolution performed to obtain the measurements from psfs
            and sources

    Optional:
        meas_size (tuple): 2d tuple of size of the detector array

    Returns:
        ndarray that is the noisy version of the input
    """
    assert sources.shape[0] == psfs.psfs.shape[1], "source and psf dimensions do not match"
    [p,_,aa,bb] = sources.shape
    [k,p,ss,ss] = psfs.psfs.shape
    ta, tb = [aa + ss - 1, bb + ss - 1]


    # FIXME: make it work for 2D input, remove selected_psfs
    # FIXME: ;move psf_dft computation to PSFs (make PSFs accept sampling_interval and o
    # output size arguments)

    psfs.selected_psfs = np.zeros((k,p,aa,bb))

    # reshape psfs
    psfs.selected_psfs = size_equalizer(psfs.psfs, ref_size=[aa,bb])

    psfs.selected_psfs = np.repeat(psfs.selected_psfs, psfs.copies.astype(int), axis=0)
    psfs.selected_psf_dfts = np.fft.fft2(psfs.selected_psfs)
    psfs.selected_psf_dfts_h = block_herm(psfs.selected_psf_dfts)
    psfs.selected_GAM = block_mul(
        psfs.selected_psf_dfts_h,
        psfs.selected_psf_dfts
    )

    # ----- forward -----
    measurement = np.fft.fftshift(
        np.fft.ifft2(
            block_mul(
                psfs.selected_psf_dfts,
                np.fft.fft2(sources)
            )
        ),
        axes=(2, 3)
    )
    return measurement.real if real else measurement


def get_contributions(real=False, *, sources, psfs):
    """
    Convolve the sources and psfs to obtain contributions.
    Args:
        sources (ndarray): 4d array of sources
        psfs (PSFs): PSFs object containing psfs and other csbs state data
        real (bool): (default=False) whether returned measurement should be real
            type of the convolution performed to obtain the measurements from psfs
            and sources

    Returns:
        ndarray that is the noisy version of the input
    """
    assert sources.shape[0] == psfs.psfs.shape[1], "source and psf dimensions do not match"

    # FIXME: make it work for 2D input
    # FIXME: ;move psf_dft computation to PSFs (make PSFs accept sampling_interval and o
    # output size arguments)

    # reshape psfs
    [p,aa,bb] = sources.shape
    [k,p,ss,ss] = psfs.psfs.shape
    psfs.psfs = size_equalizer(psfs.psfs, ref_size=[aa,bb])

    psfs.psfs = np.repeat(psfs.psfs, psfs.copies.astype(int), axis=0)
    psfs.psf_dfts = np.fft.fft2(psfs.psfs)

    # ----- forward -----
    measurement = np.fft.fftshift(
        np.fft.ifft2(
            np.einsum(
                'ijkl,jkl->ijkl',
                psfs.psf_dfts,
                np.fft.fft2(sources)
            )
        ),
        axes=(2, 3)
    )
    return measurement.real if real else measurement


@vectorize
def add_noise(signal, snr=None, max_count=None, model='Poisson', no_noise=False):
    """
    Add noise to the given signal at the specified level.

    Args:
        (ndarray): noise-free input signal
        snr (float): signal to noise ratio: for Gaussian noise model, it is
        defined as the ratio of variance of the input signal to the variance of
        the noise. For Poisson model, it is taken as the average snr where snr
        of a pixel is given by the square root of its value.
        max_count (int): Max number of photon counts in the given signal
        model (string): String that specifies the noise model. The 2 options are
        `Gaussian` and `Poisson`
        no_noise (bool): (default=False) If True, return the clean signal

    Returns:
        ndarray that is the noisy version of the input
    """
    if no_noise is True:
        return signal
    else:
        assert model.lower() in ('gaussian', 'poisson'), "invalid noise model"
        if model.lower() == 'gaussian':
            var_sig = np.var(signal)
            var_noise = var_sig / snr
            out = np.random.normal(loc=signal, scale=np.sqrt(var_noise))
        elif model.lower() == 'poisson':
            if max_count is not None:
                sig_scaled = signal * (max_count / signal.max())
                # print('SNR:{}'.format(np.sqrt(sig_scaled.mean())))
                out = poisson.rvs(sig_scaled) * (signal.max() / max_count)
            else:
                avg_brightness = snr**2
                sig_scaled = signal * (avg_brightness / signal.mean())
                out = poisson.rvs(sig_scaled) * (signal.mean() / avg_brightness)
        return out


@vectorize
def size_equalizer(x, ref_size, mode='center'):
    """
    Crop or zero-pad a 2D array so that it has the size `ref_size`.
    Both cropping and zero-padding are done such that the symmetry of the
    input signal is preserved.
    Args:
        x (ndarray): array which will be cropped/zero-padded
        ref_size (list): list containing the desired size of the array [r1,r2]
        mode (str): ('center', 'topleft') where x should be placed when zero padding
    Returns:
        ndarray that is the cropper/zero-padded version of the input
    """
    assert len(x.shape) == 2, "invalid shape for x"

    if x.shape[0] > ref_size[0]:
        pad_left, pad_right = 0, 0
        crop_left = 0 if mode == 'topleft' else (x.shape[0] - ref_size[0]) // 2
        crop_right = crop_left + ref_size[0]
    else:
        crop_left, crop_right = 0, x.shape[0]
        pad_right = ref_size[0] - x.shape[0] if mode == 'topleft' else (ref_size[0] - x.shape[0]) // 2
        pad_left = ref_size[0] - pad_right - x.shape[0]
    if x.shape[1] > ref_size[1]:
        pad_top, pad_bottom = 0, 0
        crop_top = 0 if mode == 'topleft' else (x.shape[1] - ref_size[1]) // 2
        crop_bottom = crop_top + ref_size[1]
    else:
        crop_top, crop_bottom = 0, x.shape[1]
        pad_bottom = ref_size[1] - x.shape[1] if mode == 'topleft' else (ref_size[1] - x.shape[1]) // 2
        pad_top = ref_size[1] - pad_bottom - x.shape[1]

    # crop x
    cropped = x[crop_left:crop_right, crop_top:crop_bottom]
    # pad x
    padded = np.pad(
        cropped,
        ((pad_left, pad_right), (pad_top, pad_bottom)),
        mode='constant'
    )

    return padded


def downsample(x, factor=2):
    """
    Downsample an image by average factor*factor sized patches.  Discards remaining pixels
    on bottom and right edges

    Args:
        x (ndarray): input image to downsample
        factor (int): factor to downsample image by

    Returns:
        ndarray containing downsampled image
    """

    return fftconvolve(
        x,
        np.ones((factor, factor)) / factor**2,
        mode='valid'
    )[::factor, ::factor]

def upsample(x, factor=2):
    """
    Upsample an image by turning 1 pixel into a factor*factor sized patch

    Args:
        x (ndarray): input image to upsample
        factor (int): factor to upsample image by

    Returns:
        ndarray containing upsampled image
    """

    return np.repeat(
        np.repeat(
            x,
            repeats=factor,
            axis=0
        ),
        repeats=factor,
        axis=1
    )
