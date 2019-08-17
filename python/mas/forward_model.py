#!/usr/bin/env python3
# Ulas Kamaci, Evan Widloski - 2018-08-27

import numpy as np
from mas.block import block_mul, block_herm
from mas.decorators import vectorize, _vectorize
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

@_vectorize(signature='(i,j,k)->(m,n,o)', included=[0, 'sources'])
def get_measurements(*, sources, psfs, mode='valid', real=True, meas_size=None, **kwargs):
    """
    Convolve the sources and psfs to obtain measurements.
    Args:
        sources (ndarray): 4d array of sources
        psfs (PSFs): PSFs object containing psfs and other csbs state data
        mode (string): {'circular', 'valid'} (default='valid') convolution mode.
            `circular`: circular convolution of the source and the psfs. `valid`:
            linearly convolve the sources and psfs, then take the fully overlapping
            part.
        real (bool): (default=True) whether returned measurement should be real
            type of the convolution performed to obtain the measurements from psfs
            and sources

    Optional:
        meas_size (tuple): 2d tuple of size of the detector array

    Returns:
        ndarray that is the noisy version of the input
    """
    if mode == 'circular':
        if meas_size is not None:
            sources = size_equalizer(sources, ref_size=meas_size)

        # reshape psfs and sources
        psfs_ext = np.repeat(
            size_equalizer(psfs.psfs, ref_size=meas_size),
            psfs.copies.astype(int),axis=0
        )

        # ----- forward -----
        measurement = np.fft.fftshift(
            np.fft.ifft2(
                np.einsum(
                    'ijkl,jkl->ikl',
                    np.fft.fft2(psfs_ext),
                    np.fft.fft2(sources)
                )
            ),
            axes=(1, 2)
        )

    if mode == 'valid':
        [p, aa, bb] = sources.shape
        [k, p, ss, ss] = psfs.psfs.shape
        if meas_size is not None:
            [ma, mb] = meas_size
            sources = size_equalizer(sources, ref_size=[ma + ss - 1, mb + ss - 1])
            [p, aa, bb] = sources.shape
            ta, tb = [aa + ss - 1, bb + ss - 1]
            fa, fb = [aa - ss + 1, bb - ss + 1]
        else:
            ta, tb = [aa + ss - 1, bb + ss - 1]
            fa, fb = [aa - ss + 1, bb - ss + 1]

        # reshape psfs and sources
        psfs_ext = np.repeat(
            size_equalizer(psfs.psfs, ref_size=[ta,tb]),
            psfs.copies.astype(int),axis=0
        )
        sources_ext = size_equalizer(sources, ref_size=[ta,tb])

        # ----- forward -----
        measurement = np.fft.fftshift(
            np.fft.ifft2(
                np.einsum(
                    'ijkl,jkl->ikl',
                    np.fft.fft2(psfs_ext),
                    np.fft.fft2(sources_ext)
                )
            ),
            axes=(1, 2)
        )
        measurement = size_equalizer(measurement, ref_size=[fa,fb])

    return measurement.real if real else measurement


def get_contributions(real=True, *, sources, psfs):
    """
    Convolve the sources and psfs to obtain contributions.
    Args:
        sources (ndarray): 4d array of sources
        psfs (PSFs): PSFs object containing psfs and other csbs state data
        real (bool): (default=True) whether returned measurement should be real
            type of the convolution performed to obtain the measurements from psfs
            and sources

    Returns:
        ndarray that is the noisy version of the input
    """
    assert sources.shape[0] == psfs.psfs.shape[1], "source and psf dimensions do not match"

    # reshape psfs
    [p,aa,bb] = sources.shape
    [k,p,ss,ss] = psfs.psfs.shape
    psfs = size_equalizer(psfs.psfs, ref_size=[aa,bb])

    psfs = np.repeat(psfs, psfs.copies.astype(int), axis=0)
    psf_dfts = np.fft.fft2(psfs)

    # ----- forward -----
    measurement = np.fft.fftshift(
        np.fft.ifft2(
            np.einsum(
                'ijkl,jkl->ijkl',
                psf_dfts,
                np.fft.fft2(sources)
            )
        ),
        axes=(2, 3)
    )
    return measurement.real if real else measurement


@vectorize
def add_noise(signal, dbsnr=None, max_count=None, model='Poisson', no_noise=False):
    """
    Add noise to the given signal at the specified level.

    Args:
        (ndarray): noise-free input signal
        dbsnr (float): signal to noise ratio in dB: for Gaussian noise model, it is
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
            var_noise = var_sig / 10**(dbsnr / 10)
            out = np.random.normal(loc=signal, scale=np.sqrt(var_noise))
        elif model.lower() == 'poisson':
            if max_count is not None:
                sig_scaled = signal * (max_count / signal.max())
                # print('SNR:{}'.format(np.sqrt(sig_scaled.mean())))
                out = poisson.rvs(sig_scaled) * (signal.max() / max_count)
            else:
                avg_brightness = 10**(dbsnr / 10)**2
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
        crop_left = 0 if mode == 'topleft' else (x.shape[0] - ref_size[0] + 1) // 2
        crop_right = crop_left + ref_size[0]
    else:
        crop_left, crop_right = 0, x.shape[0]
        pad_right = ref_size[0] - x.shape[0] if mode == 'topleft' else (ref_size[0] - x.shape[0]) // 2
        pad_left = ref_size[0] - pad_right - x.shape[0]
    if x.shape[1] > ref_size[1]:
        pad_top, pad_bottom = 0, 0
        crop_top = 0 if mode == 'topleft' else (x.shape[1] - ref_size[1] + 1) // 2
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

def size_compressor(signal, energy_ratio=0.9995):
    """
    Crop the borders of an image such that the given `energy_ratio` is achieved

    Args:
        signal (ndarray): input image to crop
        energy_ratio (float): ratio of output to input signal energies

    Returns:
        ndarray containing cropped image
    """

    energy0 = sum(sum(signal**2))
    energy_target = energy0 * energy_ratio
    width = int(signal.shape[0] / 2) + 1
    width_change = signal.shape[0] - width
    while width_change > 1:
        energy2 = sum(sum((size_equalizer(signal,[width,width]))**2))
        width_change = int(width_change / 2)
        if energy2 >= energy_target:
            width -= width_change
        else:
            width += width_change
    width = int(width / 2) * 2 + 1
    return width

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
