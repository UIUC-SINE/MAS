import numpy as np
from itertools import combinations
from skimage.feature.register_translation import _upsampled_dft, _compute_error, _compute_phasediff

def register_translation(src_images, target_images, upsample_factor=1,
                         space="real"):
    """
    Efficient subpixel image translation registration by cross-correlation.

    This code gives the same precision as the FFT upsampled cross-correlation
    in a fraction of the computation time and with reduced memory requirements.
    It obtains an initial estimate of the cross-correlation peak by an FFT and
    then refines the shift estimation by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT.

    Parameters
    ----------
    src_image : ndarray
        Reference image.
    target_image : ndarray
        Image to register.  Must be same dimensionality as ``src_image``.
    upsample_factor : int, optional
        Upsampling factor. Images will be registered to within
        ``1 / upsample_factor`` of a pixel. For example
        ``upsample_factor == 20`` means the images will be registered
        within 1/20th of a pixel.  Default is 1 (no upsampling)
    space : string, one of "real" or "fourier", optional
        Defines how the algorithm interprets input data.  "real" means data
        will be FFT'd to compute the correlation, while "fourier" data will
        bypass FFT of input data.  Case insensitive.

    Returns
    -------
    shifts : ndarray
        Shift vector (in pixels) required to register ``target_image`` with
        ``src_image``.  Axis ordering is consistent with numpy (e.g. Z, Y, X)
    error : float
        Translation invariant normalized RMS error between ``src_image`` and
        ``target_image``.
    phasediff : float
        Global phase difference between the two images (should be
        zero if images are non-negative).

    References
    ----------
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           "Efficient subpixel image registration algorithms,"
           Optics Letters 33, 156-158 (2008). DOI:10.1364/OL.33.000156
    .. [2] James R. Fienup, "Invariant error metrics for image reconstruction"
           Optics Letters 36, 8352-8357 (1997). DOI:10.1364/AO.36.008352
    """
    # images must be the same shape
    if src_images.shape != target_images.shape:
        raise ValueError("Error: images must be same size for "
                         "register_translation")

    # only 2D data makes sense right now
    if src_images[0].ndim != 2 and upsample_factor > 1:
        raise NotImplementedError("Error: register_translation only supports "
                                  "subpixel registration for 2D images")

    # real data needs to be fft'd.
    src_freqs = np.fft.fftn(src_images, axes=(1, 2))
    target_freqs = np.fft.fftn(target_images, axes=(1, 2))

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freqs[0].shape
    image_products = []
    time_diffs = []
    weights = []
    for i, j in combinations(range(1, len(src_freqs) - 1), 2):
        image_products.append(src_freqs[i] * target_freqs[j].conj())
        time_diffs.append(j - i)
        import ipdb
        ipdb.set_trace()
    image_products = np.array(image_products)
    time_diffs = np.array(time_diffs)

    cross_correlations = np.fft.ifftn(image_products, axes=(1, 2))

    # Locate maximum
    for time_diff in range(1, len(src_freqs) - 1):
        maxima = np.unravel_index(
            np.argmax(np.abs(cross_correlations[
                np.where(time_diffs == time_diff)
            ]
            ).sum(axis=0)),
            cross_correlations[0].shape
        )
        import ipdb
        ipdb.set_trace()
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    if upsample_factor == 1:
        src_amp = np.sum(np.abs(src_freqs[0]) ** 2) / src_freqs[0].size
        target_amp = np.sum(np.abs(target_freqs[0]) ** 2) / target_freqs[0].size
        CCmax = cross_correlations.max()
    # If upsampling > 1, then refine estimate with matrix multiply DFT
    else:
        # Initial shift estimate in upsampled grid
        shifts = np.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.0)
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        normalization = (src_freqs[0].size * upsample_factor ** 2)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts*upsample_factor
        cross_correlations = []
        for image_product in image_products:
            cross_correlation = _upsampled_dft(
                image_product.conj(),
                upsampled_region_size,
                upsample_factor,
                sample_region_offset
            ).conj()
            cross_correlation /= normalization
            cross_correlations.append(cross_correlation)
        cross_correlations = np.array(cross_correlations)
        # Locate maximum and map back to original pixel grid
        maxima = np.array(np.unravel_index(
                              np.argmax(np.sum(np.abs(cross_correlations), axis=0)),
                              cross_correlation.shape),
                          dtype=np.float64)
        maxima -= dftshift
        shifts = shifts + maxima / upsample_factor
        # import ipdb
        # ipdb.set_trace()
        CCmax = cross_correlation.max()
        src_amp = _upsampled_dft(src_freqs[0] * src_freqs[0].conj(),
                                 1, upsample_factor)[0, 0]
        src_amp /= normalization
        target_amp = _upsampled_dft(target_freqs[0] * target_freqs[0].conj(),
                                    1, upsample_factor)[0, 0]
        target_amp /= normalization

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freqs[0].ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    return shifts, _compute_error(CCmax, src_amp, target_amp),\
        _compute_phasediff(CCmax)
