#!/usr/bin/env python3
# Ulas Kamaci, Evan Widloski - 2018-08-27

import numpy as np
from mas.block import block_mul, block_herm
from mas.decorators import vectorize, _vectorize
from scipy.stats import poisson
from scipy.signal import fftconvolve
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
import logging
from mas.psf_generator import PSFs, PhotonSieve
from mas.strand_generator import strands
from mas.tracking import mb_kernel
from skimage.transform import resize, rescale
import torch
from torch.nn.functional import conv2d

log = logging.getLogger(name=__name__)

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
def get_measurements(
    *,
    sources,
    psfs,
    mode='circular',
    real=True,
    meas_size=None,
    blur_sigma=None,
    noise_sigma=None,
    drift_amount=None,
    **kwargs
):
    """
    Convolve the sources and psfs to obtain measurements.
    Args:
        sources (ndarray): 4d array of sources
        psfs (PSFs): PSFs object containing psfs and other csbs state data
        mode (string): {'circular', 'valid'} (default='valid') convolution mode.
            `circular`: circular convolution of the source and the psfs. `valid`:
            linearly convolve the sources and psfs, then take the fully overlapping
            part.  `auto`: use `valid` if possible, then fall back to `circular`
        real (bool): (default=True) whether returned measurement should be real
            type of the convolution performed to obtain the measurements from psfs
            and sources

    Optional:
        meas_size (tuple): 2d tuple of size of the detector array
        blur_sigma (float): std deviation of additional blur applied to the PSFs
        noise_sigma (float): std deviation of additional noise applied to the PSFs

    Returns:
        ndarray that is the noisy version of the input
    """

    if mode == 'auto':
        _, source_width, _ = sources.shape
        _, _, psf_width, _ = psfs.psfs.shape

        if source_width > psf_width:
            mode = 'valid'
        else:
            mode = 'circular'
            log.warning('Falling back to circular convolution')

    if mode == 'circular':
        [p, aa, bb] = sources.shape
        [k, p, ss, ss] = psfs.psfs.shape
        if meas_size is not None:
            sources = size_equalizer(sources, ref_size=meas_size)
            [p, aa, bb] = sources.shape

        # reshape psfs and sources
        psfs_ext = np.repeat(
            size_equalizer(psfs.psfs, ref_size=[aa,bb]),
            psfs.copies.astype(int),axis=0
        )

        if blur_sigma is not None:
            for i in range(psfs_ext.shape[0]):
                for j in range(psfs_ext.shape[1]):
                    psfs_ext[i,j] = gaussian_filter(psfs_ext[i,j], sigma=blur_sigma)
                    psfs.psfs_modified = size_equalizer(psfs_ext, ref_size=[ss,ss])

        if drift_amount is not None:
            kernel = np.zeros((drift_amount,drift_amount))
            for i in range(drift_amount):
                for j in range(drift_amount):
                    if i+j == drift_amount - 1:
                        kernel[i,j] = 1 / drift_amount
            for i in range(psfs_ext.shape[0]):
                for j in range(psfs_ext.shape[1]):
                    psfs_ext[i,j] = convolve2d(
                        psfs_ext[i,j], kernel, mode='same')
            psfs.psfs_modified = size_equalizer(psfs_ext, ref_size=[ss,ss])


        if noise_sigma is not None:
            psfs_ext = np.random.normal(loc=psfs_ext, scale=noise_sigma)
            psfs_ext[psfs_ext < 0] = 0
            psfs.psfs_modified = size_equalizer(psfs_ext, ref_size=[ss,ss])


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

        if blur_sigma is not None:
            for i in range(psfs_ext.shape[0]):
                for j in range(psfs_ext.shape[1]):
                    psfs_ext[i,j] = gaussian_filter(psfs_ext[i,j], sigma=blur_sigma)
                    psfs.psfs_modified = size_equalizer(psfs_ext, ref_size=[ss,ss])

        if drift_amount is not None:
            kernel = np.zeros((drift_amount,drift_amount))
            for i in range(drift_amount):
                for j in range(drift_amount):
                    if i+j == drift_amount - 1:
                        kernel[i,j] = 1 / drift_amount
            for i in range(psfs_ext.shape[0]):
                for j in range(psfs_ext.shape[1]):
                    psfs_ext[i,j] = convolve2d(
                        psfs_ext[i,j], kernel, mode='same')
            psfs.psfs_modified = size_equalizer(psfs_ext, ref_size=[ss,ss])

        if noise_sigma is not None:
            psfs_ext = np.random.normal(loc=psfs_ext, scale=noise_sigma)
            psfs_ext[psfs_ext < 0] = 0
            psfs.psfs_modified = size_equalizer(psfs_ext, ref_size=[ss,ss])

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


def get_fwd_op_torch(
    diameter=75e-3, # meters
    smallest_hole_diameter=16e-6, # meters
    wavelengths=np.array([30.4e-9]), # meters
    plane_offset=15e-3, # meters
    drift_angle=-45, # degrees
    drift_velocity=0.10e-3, # meters / s
    jitter_rms=3e-6, # meters
    frame_rate=7.5, # Hz
    pixel_size=7e-6, # meters
    device=None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ps = PhotonSieve(
        diameter=diameter,
        smallest_hole_diameter=smallest_hole_diameter
        )
    psfs = PSFs(
        ps,
        sampling_interval=pixel_size,
        source_wavelengths=wavelengths,
        measurement_wavelengths=wavelengths,
        plane_offset=plane_offset
    )

    true_drift = drift_velocity / frame_rate * np.array([
        -np.sin(np.deg2rad(drift_angle)), # use image coordinate system
        np.cos(np.deg2rad(drift_angle))
    ]) / pixel_size

    ker, ker_um = mb_kernel(true_drift, pixel_size*1e6)
    ker_um = size_equalizer(ker_um, (31,31))
    psf_mb = rescale(
        size_equalizer(gaussian_filter(ker_um, sigma=jitter_rms*1e6), (77,77)),
        1./7,
        anti_aliasing=True
    )

    psfs.psfs[0,0] /= psfs.psfs[0,0].sum()
    psf_mb /= psf_mb.sum()

    psf_final = convolve2d(psf_mb, psfs.psfs[0,0], mode='same')
    psf_final /= psf_final.sum()

    psf_final = torch.tensor(psf_final).to(device).view(1, 1, *psf_final.shape)

    def fwd_op_torch(x):
        if type(x) is not torch.Tensor:
            x = torch.tensor(x).to(device)
        x = x.view((1,) * (4 - x.dim()) + x.shape).to(device) # make sure x is 4D
        return conv2d(x, psf_final, padding='same')

    return fwd_op_torch, psf_final


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


def crop(im, center=None, *, width):
    """
    Return a cropped rectangle from an input image

    Args:
        im (ndarray): input image
        center (tuple): coordinate pair of center of cropped rectangle. defaults to image center
        width (int, tuple): length of each axis of cropped rectangle.  returns square if integer

    Returns:
        cropped rectangle of input image
    """

    if type(width) is int:
        width = (width, width)

    if center is None:
        center = (im.shape[0] // 2, im.shape[1] // 2)

    # assert (
    #     (0 <= center[0] - width[0]) and
    #     (0 <= center[1] - width[1]) and
    #     (im.shape[0] >= center[0] + width[0]) and
    #     (im.shape[1] >= center[1] + width[1])
    # ), "Cropped region falls outside image bounds"

    crop_left = (im.shape[0] - width[0] + 1) // 2
    crop_right = crop_left + width[0]
    crop_top = (im.shape[1] - width[1] + 1) // 2
    crop_bottom = crop_top + width[1]

    return im[crop_left:crop_right, crop_top:crop_bottom]


def size_compressor(signal, energy_ratio=0.9995):
    """
    Compute width of an image such that the given `energy_ratio` is achieved

    Args:
        signal (ndarray): 2D input image
        energy_ratio (float): ratio of output to input signal energies

    Returns:
        width containing 'energy_ratio' percent of total energy, measured
        from the central pixel
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
    Downsample an image by averaging factor*factor sized patches.  Discards remaining pixels
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

def downsample_mid(x, factor=2):
    """
    Downsample an image by average factor*factor sized patches.
    Shift patches by factor / 2

    Args:
        x (ndarray): input image to downsample
        factor (int): factor to downsample image by

    Returns:
        ndarray containing downsampled image
    """

    X = np.fft.fftn(x)
    kern = size_equalizer(np.ones((factor, factor)) / factor**2, x.shape)
    kern = np.fft.fftshift(kern)
    kern = np.fft.fftn(kern)
    # -factor to prevent overlap from first kernel pos to last kernel pos
    # return np.fft.ifftn(kern * X)[:-factor:factor, :-factor:factor]
    return np.fft.ifftn(kern * X)[::factor, ::factor]

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

def dof2wavelength(*, dof, base_wavelength, ps):
    """Returns a wavelength relative to base_wavelength, measured in DOF
    for a given photon sieve

    Args:
        dof (float, ndarray): dof(s) of returned wavelength relative
            to base_wavelength
        base_wavelength (float): reference wavelength
        ps (PhotonSieve): photon sieve parameters to compute DOF
    """

    return (
        base_wavelength + 2 * ps.smallest_hole_diameter *
        base_wavelength * dof / ps.diameter
    )

def wavelength2dof(*, wavelength, base_wavelength, ps):
    """Returns dof of a wavelength relative to base_wavelength
    for a given photon sieve

    Args:
        dof (float, ndarray): wavelength(s) with which to compute DOF
        base_wavelength (float): reference wavelength
        ps (PhotonSieve): photon sieve parameters to compute DOF
    """

    return (
        (wavelength - base_wavelength) * ps.diameter /
        (2 * ps.smallest_hole_diameter * base_wavelength)
    )

def modulate(shape, amp=1, width=1, grid=5):

    tile = np.tile(
        np.hstack((
            [.5] * width,
            [-.5] * width
        )),
        (width * grid, grid // 2 + 1)
    )[:, :width * grid]

    modulated = np.vstack((
        np.hstack((tile, tile.T)),
        np.hstack((tile.T, tile)),
    ))

    modulated = np.tile(
        modulated,
        (
            shape[0] // modulated.shape[0] + 1,
            shape[1] // modulated.shape[1] + 1
        )
    )[:shape[0], :shape[1]]

    return amp * modulated
