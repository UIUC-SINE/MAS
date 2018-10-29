#!/usr/bin/env python3
# Ulas Kamaci 2018-08-27

import numpy as np
from mas.sse_cost import block_mul, block_herm
from scipy.stats import poisson

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


def get_measurements(*, sources, psfs, mode = 'circular'):
    """
    Convolve the sources and psfs to obtain measurements.
    Args:
        sources (ndarray): 4d array of sources
        psfs (PSFs): PSFs object containing psfs and other csbs state data
        mode (string): {'circular', 'linear'} (default='circular')
        type of the convolution performed to obtain the measurements from psfs
        and sources

    Returns:
        ndarray that is the noisy version of the input
    """
    assert sources.shape[0] == psfs.psfs.shape[1], "source and psf dimensions do not match"
    [p,_,aa,bb] = sources.shape
    [k,p,ss,ss] = psfs.psfs.shape
    psfs.selected_psfs = np.zeros((k,p,aa,bb))

    # reshape psfs
    for i in range(k):
        for j in range(p):
            psfs.selected_psfs[i,j,:,:] = size_equalizer(psfs.psfs[i,j,:,:], [aa,bb])

    psfs.selected_psfs = np.repeat(psfs.selected_psfs, psfs.copies.astype(int), axis=0)
    psfs.selected_psf_dfts = np.fft.fft2(psfs.selected_psfs)
    psfs.selected_psf_dfts_h = block_herm(psfs.selected_psf_dfts)

    # ----- forward -----
    return np.real(
        np.fft.ifft2(
            block_mul(
                psfs.selected_psf_dfts,
                np.fft.fft2(sources)
            )
        )
    )


def add_noise(signal, snr_db=None, exp_time=None, model='Gaussian', nonoise=False):
    """
    Add noise to the given signal at the specified level.

    Args:
        (ndarray): noise-free input signal
        snr_db (float): 10*log_10(SNR) where SNR is defined as the ratio of
        variance of the input signal to the variance of the noise. Necessary
        input for Gaussian noise.
        exp_time (float): Exposure time in seconds. Necessary input for
        Poisson noiseself.
        model (string): String that specifies the noise model. The 2 options are
        `Gaussian` and `Poisson`
        nonoise (bool): (default=False) If True, return the clean signal

    Returns:
        ndarray that is the noisy version of the input
    """
    if nonoise is True:
        return signal
    else:
        assert model == 'Gaussian' or 'Poisson', "select the model correctly"
        if model == 'Gaussian':
            assert snr_db is not None, "please specify snr_db"
            var_sig = np.var(signal)
            var_noise = var_sig / (10**(snr_db / 10))
            out = np.random.normal(loc=signal, scale=np.sqrt(var_noise))
        elif model == 'Poisson':
            assert exp_time is not None, "please specify exp_time in seconds"
            beta = 1
            out = poisson.rvs(signal * exp_time * beta)
        return out


def size_equalizer(x, ref_size):
    """
    Crop or zero-pad a 2D array so that it has the size `ref_size`.
    Both cropping and zero-padding are done such that the symmetry of the
    input signal is preserved.
    Args:
        x (ndarray): array which will be cropped/zero-padded
        ref_size (list): list containing the desired size of the array [r1,r2]
    Returns:
        ndarray that is the cropper/zero-padded version of the input
    """
    [i1, i2] = x.shape
    [r1, r2] = ref_size
    [f1, f2] = [r1 - i1, r2 - i2]
    m1 = int(i1/2)
    m2 = int(i2/2)
    down = int((r1 - 1) / 2)
    up = r1 - down - 1
    right = int((r2 - 1) / 2)
    left = r2 - right -1

    out = x

    for i,k in enumerate((f1,f2)):

        if k > 0:
            after = int(k/2)
            before = k - after
            if i == 0:
                out = np.pad(out, ((before, after), (0, 0)), mode = 'constant')
            else:
                out = np.pad(out, ((0, 0), (before, after)), mode = 'constant')

        elif k == 0:
            out = np.pad(out, ((0, 0), (0, 0)), mode = 'constant')

        elif k < 0:
            if i == 0:
                out = out[m1 - up : m1 + down + 1, :]
            else:
                out = out[:, m2 - left : m2 + right + 1]

    return out
