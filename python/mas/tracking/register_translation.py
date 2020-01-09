import numpy as np
from itertools import combinations
from skimage.feature.register_translation import _upsampled_dft, _compute_error, _compute_phasediff

def register_translation(frames, upsample_factor=1, time_diff=20):
    """
    Efficient subpixel image translation registration by cross-correlation.

    Parameters
    ----------
    frames (ndarray):
        input frames
    upsample_factor (int):
        Upsampling factor. Images will be registered to within
        ``1 / upsample_factor`` of a pixel. For example
        ``upsample_factor == 20`` means the images will be registered
        within 1/20th of a pixel.  Default is 1 (no upsampling)
    time_diff (int):
        frame separation when computing correlation

    Returns
    -------
    shifts : ndarray
        Shift vector (in pixels) required to register ``target_image`` with
        ``src_image``.  Axis ordering is consistent with numpy (e.g. Z, Y, X)

    References
    ----------
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           "Efficient subpixel image registration algorithms,"
           Optics Letters 33, 156-158 (2008). :DOI:`10.1364/OL.33.000156`
    .. [2] James R. Fienup, "Invariant error metrics for image reconstruction"
           Optics Letters 36, 8352-8357 (1997). :DOI:`10.1364/AO.36.008352`
    """

    freq = np.fft.fftn(frames, axes=(1, 2))

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    # shape = src_freq.shape
    shape = freq[0].shape
    # image_product = src_freq * target_freq.conj()
    image_products = freq[:-time_diff] * freq[time_diff:].conj()
    image_product = np.sum(image_products, axis=0)
    cross_correlation = np.fft.ifftn(image_product)

    # Locate maximum
    maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)),
                              cross_correlation.shape)
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    # Initial shift estimate in upsampled grid
    shifts = np.round(shifts * upsample_factor) / upsample_factor
    upsampled_region_size = np.ceil(upsample_factor * 1.5)
    # Center of output array at dftshift + 1
    dftshift = np.fix(upsampled_region_size / 2.0)
    upsample_factor = np.array(upsample_factor, dtype=np.float64)
    # normalization = (src_freq.size * upsample_factor ** 2)
    # Matrix multiply DFT around the current shift estimate
    sample_region_offset = dftshift - shifts*upsample_factor
    cross_correlation = _upsampled_dft(
        image_product.conj(),
        upsampled_region_size,
        upsample_factor,
        sample_region_offset
    ).conj()
    # cross_correlation /= normalization
    # Locate maximum and map back to original pixel grid
    maxima = np.unravel_index(
        np.argmax(np.abs(cross_correlation)),
        cross_correlation.shape
    )
    CCmax = cross_correlation[maxima]

    maxima = np.array(maxima, dtype=np.float64) - dftshift

    shifts = shifts + maxima / upsample_factor

    return shifts
