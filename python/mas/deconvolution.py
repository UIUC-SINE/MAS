#!/usr/bin/env python3
# Ulas Kamaci 2018-08-27

import numpy as np
import logging, itertools, functools, multiprocessing
from matplotlib import pyplot as plt
from mas.sse_cost import block_mul, block_inv, block_herm, SIG_e_dft, get_LAM
from mas.forward_model import size_equalizer
from scipy.interpolate import interp1d
from scipy.fftpack import dct, dctn, idctn
from mas.plotting import plotter4d
from skimage.measure import compare_ssim as ssim
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(message)s')

def patch_extractor(image, *, patch_shape):
    """Create a patch matrix where each column is a vectorized patch.
    It works with both 2D and 3D patches. Patches at the boundaries are
    extrapolated as if the image is periodically replicated. This way all
    the patches have the same dimension.

    Args:
        image (ndarray): 4d array of spectral images
        patch_shape (tuple): tuple of length 3: (dim_x, dim_y, dim_z)

    Returns:
        patch_mtx (ndarray): patch matrix containing vectorized patches in its
        columns. Number of columns (patches) is equal to the number of pixels
        (or voxels) in the image.
    """
    assert len(patch_shape) == 3, 'patch_shape must have length=3'

    [p,_,aa,bb] = image.shape
    patch_size = np.size(np.empty(patch_shape))
    patch_mtx = np.zeros((patch_size, np.size(image)))

    # periodically extend the input image
    temp = np.concatenate((image, image[:patch_shape[2] - 1,:,:,:]), axis = 0)
    temp = np.concatenate((temp, temp[:,:,:patch_shape[0] - 1,:]), axis = 2)
    temp = np.concatenate((temp, temp[:,:,:,:patch_shape[1] - 1]), axis = 3)
    [rows, cols, slices] = np.unravel_index(
        range(patch_size), patch_shape
    )
    for i in range(patch_size):
        patch_mtx[i,:] = np.reshape(
            temp[
                slices[i] : p + slices[i],
                :,
                rows[i] : aa + rows[i],
                cols[i] : bb + cols[i],
            ],
            -1
        )

    return patch_mtx


def patch_aggregator(patch_mtx, *, patch_shape, image_shape):
    """Implements the adjoint of the patch extractor operator.

    Args:
        patch_mtx (ndarray): patch matrix containing vectorized patches in its
        columns. Number of columns (patches) is equal to the number of pixels
        (or voxels) in the image.
        patch_shape (tuple): tuple of length 3: (dim_x, dim_y, dim_z)
        image_shape (tuple): tuple of length 3: (dim_z, 1, dim_x, dim_y)

    Returns:
        image (ndarray): 3D matrix consisting of the aggregated patches
    """
    temp = np.zeros(
        (image_shape[0]+patch_shape[2] - 1,) +
        (1,) +
        (image_shape[2]+patch_shape[0] - 1,) +
        (image_shape[3]+patch_shape[1] - 1,)
    )

    [rows, cols, slices] = np.unravel_index(
    range(patch_mtx.shape[0]), patch_shape)

    for i in range(patch_mtx.shape[0]):
        temp[
            slices[i] : image_shape[0] + slices[i],
            :,
            rows[i] : image_shape[2] + rows[i],
            cols[i] : image_shape[3] + cols[i]
        ] += np.reshape(
            patch_mtx[i,:], image_shape
        )

    temp[:patch_shape[2] - 1,:,:,:] += temp[image_shape[0]:,:,:,:]
    temp[:,:,:patch_shape[0] - 1,:] += temp[:,:,image_shape[2]:,:]
    temp[:,:,:,:patch_shape[1] - 1] += temp[:,:,:,image_shape[3]:]

    return temp[:image_shape[0],:,:image_shape[2],:image_shape[3]]

def sparse_coding(recon, *, patches, transform):
    """Perform sparse coding operation on the image patches.

    Takes the patches in the image domain, multiplies with the sparsifying
    transform, and then does the specified type of thresholding.

    Args:
        recon (Reconstruction): Reconstruction object containing the threshold
            and the type of thresholding
        patches (ndarray): 2d array of vectorized patches in each column.
        transform (ndarray): 2d array of the sparsifying transform

    Returns:
        ndarray: thresholded patches in the transform domain
    """

    sparse_codes = transform @ patches
    if hasattr(recon, 'sparsity_ratio'): # constrained
        sparsity_threshold = np.sort(
            abs(sparse_codes), axis=None
        )[int(np.size(patches) * recon.sparsity_ratio)]
    else: # unconstrained
        sparsity_threshold = recon.sparsity_threshold

    sparse_indices = (sparse_codes > sparsity_threshold).astype(np.int)
    if not sparse_indices.any():
        # return sparse_codes
        return np.zeros_like(sparse_codes)

    return sparse_codes * sparse_indices

def dctmtx(shape):
    """Return the DCT matrix

    Args:
        shape (tuple): original shape of the flattened array that the DCT matrix
        operates on

    Returns:
        ndarray of the DCT matrix
    """
    def dctmtx1d(size):
        mtx = np.zeros((size,size))
        for i in range(size):
            a = np.zeros(size)
            a[i] = 1
            mtx[:,i] = dct(a, norm='ortho')
        return mtx

    if type(shape) is int:
        return dctmtx1d(shape)

    mtx = dctmtx1d(shape[0])
    for s in shape[1:]:
        mtx = np.kron(dctmtx1d(s), mtx)

    return mtx

def bccb_mtx(x):
    """Takes the reshaped first column, returns the BCCB matrix

    Args:
        x (ndarray): 2d array which is the reshaped version of the first column
        of the BCCB matrix

    Returns:
        ndarray of the BCCB matrix
    """
    [a,b] = x.shape
    mtx = np.zeros((a*b,a*b))

    for i in range(a*b):
        ii = int(i/a)
        jj = i % b
        z = np.roll(x, ii, axis=0)
        z = np.roll(z, jj, axis=1)
        mtx[:,i] = np.reshape(z, (a*b,))
    return mtx

def indsum(x, y, indices):
    """Add the unordered patches in your patch matrix to the ordered one.

    Input matrix `y` has unordered patches in its columns (there may be repeating
    patches). This function aggragates the patches in `y` on `x` specified by
    the `indices`. The fact that there may be repeating patches made this
    function necessary, otherwise the implementation is one line.

    Args:
        x (ndarray): ordered patch matrix on which the aggregation will occur
        y (ndarray): unordered patch patrix whose patches will be added to `x`
        indices (ndarray): 1d array holding the indices of patches in `y`
            (where they belong to in `x`)
    Returns:
        x (ndarray)
    """
    arg_old = np.arange(len(indices))
    ind_old = indices
    while len(arg_old) > 0:
        ind_new, arg_new = np.unique(ind_old, return_index=True)
        arg_new = arg_old[arg_new]
        x[:, ind_new] += y[:, arg_new]
        arg_old = np.array(list((Counter(arg_old) - Counter(arg_new)).keys()), dtype=np.int)
        ind_old = indices[arg_old]
    return x

def lowrank(i, patches_zeromean, window_size, imsize, threshold, group_size):
    """Form a group of similar patches to `i`th patch, and take the lowrank
    approximation of the group with the specified thresholding type and value.

    Since this function is computationally expensive, the similar patches are
    searched only in a window around the given patch, whose size is specified
    by `window_size`. Euclidean distance is used as the similarity metric.

    Args:
        i (int): index of the patch of interest
        patches_zeromean (ndarray): 2d array of zero mean patches
        window_size (tuple): tuple of size of the window in which the patches are searched
        imsize (tuple): length 2 tuple of size of the reconstructed image
        threshold (tuple): thresholds to be applied on the singular values of the group matrix
        group_size (int): number of patches in the group to be formed

    Returns:
        ndarray: low-rank approximation of the matrix of the grouped patches
        ind (ndarray): array of indices of selected patches that are close to
            the `i`th patch
    """
    # find out which image the index i correspond to (to apply threshold accordingly)
    im = np.int(i / (imsize[0]*imsize[1]))

    # get the indices inside the window
    ind_wind = ind_selector(i, imsize=imsize, window_size=window_size)

    ind = ind_wind[
        np.argsort(
            np.linalg.norm(patches_zeromean[ind_wind] - patches_zeromean[i], axis=1)
        )[:group_size]
    ]
    u, s, v_T = np.linalg.svd(patches_zeromean[ind].T, full_matrices=False)
    return u @ np.diag(hard_thresholding(s, threshold=threshold[im])) @ v_T, ind


def ind_selector(i, *, imsize, window_size):
    """Given the shape of a 2d array and an index on that array, return the
    closest set of indices confined in a rectangular area of `window_size`.

    Args:
        i (int): index of the pixel of interest
        window_size (tuple): tuple of size of the window
        imsize (tuple): length 2 tuple of size of the reconstructed image

    Returns:
        ndarray: 1d array of desired indices
    """
    indo = np.zeros(2, dtype=np.int)
    aa, bb = imsize
    im = np.int(i / (aa*bb))
    i1 = i - im * aa*bb
    ind = np.unravel_index(i1, (aa, bb))
    for j in range(2):
        if ind[j] - window_size[j]/2 < 0:
            indo[j] = 0
        elif ind[j] + window_size[j]/2 > imsize[j]:
            indo[j] = imsize[j] - window_size[j]
        else:
            indo[j] = ind[j] - window_size[j]/2

    indx0 = np.kron(
        np.arange(indo[0], indo[0] + window_size[0]),
        np.ones(window_size[1], dtype=np.int)
    )
    indx1 = np.kron(
        np.ones(window_size[0], dtype=np.int),
        np.arange(indo[1], indo[1] + window_size[1])
    )

    return bb*indx0 + indx1 + im * aa*bb


# def lcurve(
#         recon,
#         *,
#         reg_param,
#         measurements,
#         sources,
#         psfs,
#         instance,
#         **kwargs
# ):
#

def difftv(p,q):
    """Finite difference operator defined as L(p,q) in doi:10.1109/tip.2009.2028250

    Only difference is that we have a multi-source implementation, so the array
    sizes are 4d instead of 2d.

    Args:
        p (ndarray): 4d array of size (num_sources, 1, aa-1, bb)
        q (ndarray): 4d array of size (num_sources, 1, aa, bb-1)

    Returns:
        x (ndarray): 4d array of size (num_sources, 1, aa, bb)
    """
    [num_sources,_,aa,bb] = p.shape
    aa += 1
    x = np.zeros((num_sources,1,aa,bb))
    for i in range(num_sources):
        tempx = np.zeros((aa,bb))
        tempx[:-1, :] = p[i, 0].copy()
        tempx[1:, :] -= p[i, 0]
        tempx[0, :] = p[i, 0, 0, :]

        tempy = np.zeros((aa,bb))
        tempy[:, :-1] = q[i, 0].copy()
        tempy[:, 1:] -= q[i, 0]
        tempy[:, 0] = q[i, 0, :, 0]

        x[i,0] = tempx + tempy

    return x

def difftv_T(x):
    """Adjoint of the finite difference operator defined as L^T(x) in
    doi:10.1109/TIP.2009.2028250

    Only difference is that we have a multi-source implementation, so the array
    sizes are 4d instead of 2d.

    Args:
        x (ndarray): 4d array of size (num_sources, 1, aa, bb)

    Returns:
        p (ndarray): 4d array of size (num_sources, 1, aa-1, bb)
        q (ndarray): 4d array of size (num_sources, 1, aa, bb-1)
    """
    [num_sources,_,aa,bb] = x.shape
    p = np.zeros((num_sources,1,aa-1,bb))
    q = np.zeros((num_sources,1,aa,bb-1))

    for i in range(num_sources):
        p[i, 0] = x[i, 0, :-1, :] - x[i, 0, 1:, :]
        q[i, 0] = x[i, 0, :, :-1] - x[i, 0, :, 1:]

    return p, q

def proj_unitball(p,q, tv='aniso'):
    """Projection onto set P defined as P_P(p,q) in doi:10.1109/TIP.2009.2028250

    Only difference is that we have a multi-source implementation, so the array
    sizes are 4d instead of 2d.

    Args:
        p (ndarray): 4d array of size (num_sources, 1, aa-1, bb)
        q (ndarray): 4d array of size (num_sources, 1, aa, bb-1)
        tv (str): {'iso', 'aniso'} (default: 'aniso')
        type of the TV norm that is used

    Returns:
        r (ndarray): 4d array of size (num_sources, 1, aa-1, bb)
        s (ndarray): 4d array of size (num_sources, 1, aa, bb-1)
    """
    if tv is 'iso':
        [num_sources,_,aa,bb] = p.shape
        aa += 1
        r = np.zeros((num_sources,1,aa-1,bb))
        s = np.zeros((num_sources,1,aa,bb-1))
        norm = np.sqrt(p[:, :, :, :-1]**2 + q[:, :, :-1, :]**2)
        norm_xlast = abs(p[:, :, :, -1])
        norm_ylast = abs(q[:, :, -1, :])
        r[:, :, :, :-1] = p[:, :, :, :-1] / np.maximum(1, norm)
        r[:, :, :, -1] = p[:, :, :, -1] / np.maximum(1, norm_xlast)

        s[:, :, :-1, :] = q[:, :, :-1, :] / np.maximum(1, norm)
        s[:, :, -1, :] = q[:, :, -1, :] / np.maximum(1, norm_ylast)

    elif tv is 'aniso':
        r = p / np.maximum(1, abs(p))
        s = q / np.maximum(1, abs(q))
    return r, s

def diff(a):
    """Discrete gradient operator acting horizontally and vertically.

    Periodic boundary condition is assumed at the boundaries.

    Args:
        a (ndarray): 4d array of size (num_sources, 1, aa, bb)

    Returns:
        diff_a (ndarray): 5d array of size (2, num_sources, 1, aa, bb). First
            and second dimensions include the horizontal and vertical gradients,
            respectively.
    """
    [p,_,aa,bb] = a.shape
    diff_a = np.zeros((2,) + a.shape)
    for i in range(p):
        tempx = a[i, 0].copy()
        tempx[:, 1:] -= a[i, 0, :, :bb-1]
        tempx[:, 0] -= a[i, 0, :, -1]
        diff_a[0, i, 0] = tempx

        tempy = a[i, 0].copy()
        tempy[1:, :] -= a[i, 0, :aa-1, :]
        tempy[0, :] -= a[i, 0, -1, :]
        diff_a[1, i, 0] = tempy

    return diff_a

def diff_T(a):
    """Adjoint of the discrete gradient operator acting horizontally and vertically.

    Periodic boundary condition is assumed at the boundaries.

    Args:
        a (ndarray): 5d array of size (2, num_sources, 1, aa, bb). First and
            second dimensions include the horizontal and vertical gradients,
            respectively.

    Returns:
        ndarray: 4d array of size (num_sources, 1, aa, bb).
    """
    [_,p,_,aa,bb] = a.shape
    diff_T_a = np.zeros(a.shape)
    for i in range(p):
        tempx = a[0, i, 0].copy()
        tempx[:, :bb-1] -= a[0, i, 0, :, 1:]
        tempx[:, -1] -= a[0, i, 0, :, 0]
        diff_T_a[0, i, 0] = tempx

        tempy = a[1, i, 0].copy()
        tempy[:aa-1, :] -= a[1, i, 0, 1:, :]
        tempy[-1, :] -= a[1, i, 0, 0, :]
        diff_T_a[1, i, 0] = tempy

    return diff_T_a[0] + diff_T_a[1]

def soft_thresholding(x, *, threshold):
    """Element-wise soft thresholding function.

    Args:
        x (ndarray): ndarray of any size
        threshold (float): threshold value of the operation

    Returns:
        y (ndarray): element-wise soft thresholded version of `x`
    """
    y = x.copy()
    y[x > threshold] -= threshold
    y[x < -threshold] += threshold
    y[abs(x) <= threshold] = 0
    return y

def hard_thresholding(x, *, threshold):
    """Element-wise hard thresholding function.

    Args:
        x (ndarray): ndarray of any size
        threshold (float): threshold value of the operation

    Returns:
        y (ndarray): element-wise hard thresholded version of `x`
    """
    return x * (abs(x) > threshold).astype(np.int)

def thresholding(x, recon, **kwargs):
    """High level function that does soft/hard thresholding.

    Args:
        x (ndarray): ndarray of any size
        recon (Reconstruction): Reconstruction object containing the thresholding
            parameters and the type of thresholding

    Returns:
        y (ndarray): element-wise hard thresholded version of `x`
    """
    if kwargs['thresholding'] is 'soft':
        threshold = recon.lam / recon.nu
        return soft_thresholding(x, threshold = threshold)

    elif kwargs['thresholding'] is 'hard':
        threshold = np.sqrt(2*recon.lam / recon.nu)
        return hard_thresholding(x, threshold = threshold)

def domain_transformer(x, recon, **kwargs):
    """High level function that takes the image x and outputs its transform
    which specified by the user in the `recon` object.

    Args:
        x (ndarray): 4d array of spectral images of size (num_sources, 1, aa, bb)
        recon (Reconstruction): Reconstruction object containing the type of
        transform to apply on `x`

    Returns:
        y (ndarray): transform domain represantation of `x`
    """
    if kwargs['sparsifyer'] is 'TV':
        return diff(x)

    elif kwargs['sparsifyer'] is 'patch_based':
        recon.patches = patch_extractor(
            x, patch_shape = recon.patch_shape
        )
        return recon.transform @ recon.patches

def domain_transformer_T(x, recon, **kwargs):
    """High level function that applies the adjoint of the domain transformation
    on the transform domain image `x`.

    Args:
        x (ndarray): ndarray of transform domain signal
        recon (Reconstruction): Reconstruction object containing the type of
        transform to apply on `x`

    Returns:
        x (ndarray): adjoint applied to `x`
    """
    if kwargs['sparsifyer'] is 'TV':
        return diff_T(x)

    elif kwargs['sparsifyer'] is 'patch_based':
        return patch_aggregator(
            recon.transform.conj().T @ x,
            patch_shape = recon.patch_shape,
            image_shape = kwargs['shape']
        )

def fista_dct(
        recon,
        *,
        sources,
        psfs,
        measurements,
        **kwargs
):
    """Function that implements PSSI deconvolution with FISTA algorithm using
    Full DCT as the sparsifying transform.

    The considered cost function has the following form:
    x_hat = argmin_x { ||Ax-y||_2^2 + \lambda ||Wx||_1 } (1)
    where W is the 2D DCT transform. Instead of (1), this function solves (2):
    z_hat = argmin_z { ||AW^{-1}z-y||_2^2 + \lambda ||z||_1 } (2)
    x_hat = W z_hat
    because the proximal step of (2) has a closed form expression (soft_thresholding).

    Args:
        recon (Reconstruction): Reconstruction object containing the reconstruction
            parameters
        sources (ndarray): 4d array of sources
        measurements (ndarray): 4d array of measurements
        psfs (PSFs): PSFs object containing psfs and related data
    """
    [k,p,aa,bb] = psfs.selected_psfs.shape
    mse_inner = np.zeros((p,recon.maxiter))
    #initialize the reconstruction
    if recon.recon_init_method is 'zeros':
        x = np.zeros((p,1,aa,bb))

    elif recon.recon_init_method is 'tikhonov':
        recon_tik = Reconstruction(
            sources=sources,
            measurements=np.expand_dims(measurements, axis=0),
            psfs=psfs,
            deconvolver=tikhonov,
            **kwargs
        )
        x = recon_tik.reconstructed

    z_old = np.zeros((p,1,aa,bb))
    z_tilde = dctn(x, norm='ortho', axes=(-1,-2))
    t_old = 1

    # pre-compute some arrays for efficiency
    psfdfts_h_meas = block_mul(psfs.selected_psf_dfts_h, np.fft.fft2(measurements))

    for iter in range(recon.maxiter):
        z_new = soft_thresholding(
            z_tilde - (1/recon.nu) * dctn(
                np.real(
                np.fft.ifft2(
                    block_mul(
                        psfs.selected_GAM,
                        np.fft.fft2(idctn(z_tilde, norm='ortho', axes=(-1,-2)))
                    ) - psfdfts_h_meas
                )
                ),
                norm='ortho', axes=(-1,-2)
            ),
            threshold = recon.lam / recon.nu
        )

        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t_old**2))

        z_tilde = z_new + ((t_old - 1) / t_new) * (z_new - z_old)

        z_old = z_new.copy()
        t_old = t_new.copy()
        x = idctn(z_new, norm='ortho', axes=(-1,-2))

        mse_inner[:,iter] = np.mean(
                (sources - x)**2,
                axis=(1, 2, 3)
        )

        dfid = 0.5*(1/np.size(measurements))*np.sum(abs(
            block_mul(
            psfs.selected_psf_dfts, np.fft.fft2(x)
        ) - np.fft.fft2(measurements)
        )**2)

        l1_reg = recon.lam * np.sum(abs(z_new))

        if iter % 50 == 0:
            print(dfid+l1_reg)
            ssim1 = np.zeros(p)
            mse1 = np.mean((sources - x)**2, axis=(1, 2, 3))
            psnr1 = 20 * np.log10(np.max(sources, axis=(1,2,3))/np.sqrt(mse1))
            for i in range(p):
                ssim1[i] = ssim(sources[i,0], x[i,0],
                    data_range=np.max(x[i,0])-np.min(x[i,0]))
            plotter4d(x,
                cmap='gist_heat',
                fignum=3,
                title='Iteration: {}\n Recon. SSIM={}\n Recon. PSNR={}'.format(iter, ssim1, psnr1)
            )
            plt.pause(0.5)

    recon.mse_inner = mse_inner
    recon.reconstructed = x.copy()

def fista_tv(
        recon,
        *,
        sources,
        psfs,
        measurements,
        **kwargs
):
    """Function that implements PSSI deconvolution with FISTA algorithm using
    total variation as the sparsifying transform.

    The considered cost function has the following form:
    x_hat = argmin_{x>0} { ||Ax-y||_2^2 + \lambda ||x||_TV }
    Please see the paper doi:10.1109/tip.2009.2028250 for details.

    Args:
        recon (Reconstruction): Reconstruction object containing the reconstruction
            parameters
        sources (ndarray): 4d array of sources
        measurements (ndarray): 4d array of measurements
        psfs (PSFs): PSFs object containing psfs and related data
    """
    [k,num_sources,aa,bb] = psfs.selected_psfs.shape
    mse_inner = np.zeros((num_sources,recon.maxiter))
    cost = np.zeros(recon.maxiter+1)
    itererror = np.zeros(recon.maxiter)
    cost[0] = 0.5 * np.sum(measurements**2)
    #initialize the reconstruction
    x = np.zeros((num_sources,1,aa,bb))
    r = p_old = np.zeros((num_sources,1,aa-1,bb))
    s = q_old = np.zeros((num_sources,1,aa,bb-1))
    t_old = 1
    # pre-compute some arrays for efficiency
    psfdfts_h_meas = block_mul(
        psfs.selected_psf_dfts_h,
        np.fft.fft2(measurements)
    )

    for iter in range(recon.maxiter):
        x_old = x.copy()
        measgrad = x - (2/recon.nu) * np.real(np.fft.ifft2(
            block_mul(psfs.selected_GAM, np.fft.fft2(x)) -
            psfdfts_h_meas
        ))
        lam2 = 2*recon.lam/recon.nu
        for it in range(50):
            temp = measgrad - recon.lam * difftv(r, s)
            temp[temp < 0] = 0
            [r_temp, s_temp] = difftv_T(temp)
            [p_new, q_new] = proj_unitball(
                r + (1 / (8 * recon.lam)) * r_temp,
                s + (1 / (8 * recon.lam)) * s_temp,
                tv = recon.tv
            )

            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t_old**2))

            r = p_new + ((t_old - 1) / t_new) * (p_new - p_old)
            s = q_new + ((t_old - 1) / t_new) * (q_new - q_old)

            t_old = t_new.copy()
            p_old = p_new.copy()
            q_old = q_new.copy()

        x = measgrad - recon.lam * difftv(p_new, q_new)
        x[x < 0] = 0

        mse_inner[:,iter] = np.mean(
                (sources - x)**2,
                axis=(1, 2, 3)
        )

        dfid = 0.5*(1/np.size(measurements))*np.sum(abs(
            block_mul(
            psfs.selected_psf_dfts, np.fft.fft2(x)
        ) - np.fft.fft2(measurements)
        )**2)

        l1_reg = recon.lam * np.sum(abs(diff(x)))

        cost[iter+1] = dfid + l1_reg
        itererror[iter] = np.sum((x-x_old)**2) / np.sum(x**2)

        if iter % 50 == 0:
            print(dfid+l1_reg)
            ssim1 = np.zeros(num_sources)
            mse1 = np.mean((sources - x)**2, axis=(1, 2, 3))
            psnr1 = 20 * np.log10(np.max(sources, axis=(1,2,3))/np.sqrt(mse1))
            for i in range(num_sources):
                ssim1[i] = ssim(sources[i,0], x[i,0],
                    data_range=np.max(x[i,0])-np.min(x[i,0]))
            plotter4d(x,
                cmap='gist_heat',
                fignum=3,
                title='Iteration: {}\n Recon. SSIM={}\n Recon. PSNR={}'.format(iter, ssim1, psnr1)
            )
            plt.pause(0.5)

    recon.mse_inner = mse_inner
    recon.cost = cost
    recon.itererror = itererror
    recon.reconstructed = x.copy()



def admm_tv(
        recon,
        *,
        sources,
        psfs,
        measurements,
        **kwargs
):
    """Function that implements PSSI deconvolution with the ADMM algorithm using
    total variation as the sparsifying transform.

    The considered cost function has the following form:
    x_hat = argmin_{x} { ||Ax-y||_2^2 + \lambda ||Dx||_1 }

    Args:
        recon (Reconstruction): Reconstruction object containing the reconstruction
            parameters
        sources (ndarray): 4d array of sources
        measurements (ndarray): 4d array of measurements
        psfs (PSFs): PSFs object containing psfs and related data
    """
    [k,p,aa,bb] = psfs.selected_psfs.shape
    mse_inner = np.zeros((p,recon.maxiter))
    cost = np.zeros(recon.maxiter+1)
    itererror = np.zeros(recon.maxiter)
    cost[0] = 0.5 * np.sum(measurements**2)
    #initialize the reconstruction
    if recon.recon_init_method is 'zeros':
        primal1 = np.zeros((p,1,aa,bb))

    elif recon.recon_init_method is 'tikhonov':
        recon_tik = Reconstruction(
            sources=sources,
            measurements=np.expand_dims(measurements, axis=0),
            psfs=psfs,
            deconvolver=tikhonov,
            **kwargs
        )
        primal1 = recon_tik.reconstructed

    primal2 = diff(primal1)
    dual = primal2.copy()
    # pre-compute some arrays for efficiency
    psfdfts_h_meas = block_mul(
        psfs.selected_psf_dfts_h,
        np.fft.fft2(measurements)
    )

    imrec = block_mul(recon.SIG_inv,psfdfts_h_meas)

    for iter in range(recon.maxiter):

        primal1_old = primal1.copy()
        ###### primal1 update ######
        primal1 = np.real(
            np.fft.ifft2(imrec+
                block_mul(
                    recon.SIG_inv,
                    np.fft.fft2(recon.nu*diff_T(primal2-dual))
                )
            )
        )

        ###### primal2 update ######
        primal2 = soft_thresholding(
            diff(primal1) + dual, threshold=recon.lam/recon.nu
        )

        ##### dual update #####
        dual += (diff(primal1) - primal2)

        mse_inner[:,iter] = np.mean(
                (sources - primal1)**2,
                axis=(1, 2, 3)
        )

        dfid = 0.5*(1/np.size(measurements))*np.sum(abs(
            block_mul(
            psfs.selected_psf_dfts, np.fft.fft2(primal1)
        ) - np.fft.fft2(measurements)
        )**2)

        l1_reg = recon.lam * np.sum(abs(primal2))
        l1_reg2 = recon.lam * np.sum(abs(diff(primal1)))

        residual = 0.5*recon.nu * np.sum((diff(primal1) - primal2)**2)

        lagrange = recon.nu * np.sum(dual * (diff(primal1) - primal2))

        # print(dfid, l1_reg, lagrange, residual, dfid+l1_reg+residual+lagrange)
        cost[iter+1] = dfid + l1_reg2
        itererror[iter] = np.sum((primal1-primal1_old)**2) / np.sum(primal1**2)

        if iter % 100 == 0 or iter == recon.maxiter - 1:
            print(dfid, l1_reg2, dfid+l1_reg2)
            ssim1 = np.zeros(p)
            mse1 = np.mean((sources - primal1)**2, axis=(1, 2, 3))
            psnr1 = 20 * np.log10(np.max(sources, axis=(1,2,3))/np.sqrt(mse1))
            for i in range(p):
                ssim1[i] = ssim(sources[i,0], primal1[i,0],
                    data_range=np.max(primal1[i,0])-np.min(primal1[i,0]))
            plotter4d(primal1,
                cmap='gist_heat',
                fignum=3,
                title='Iteration: {}\n Recon. SSIM={}\n Recon. PSNR={}'.format(iter, ssim1, psnr1)
            )
            plt.pause(0.5)

    recon.mse_inner = mse_inner
    recon.dfid.append(2*dfid)
    recon.reg.append(l1_reg2/recon.lam)
    recon.cost = cost
    recon.itererror = itererror
    recon.reconstructed = primal1

def sparsepatch(
        recon,
        *,
        sources,
        psfs,
        measurements,
        **kwargs
):
    """Function that implements PSSI deconvolution with a patch based sparsifying
    transform for sparse recovery.

    P1 and P3 formulations described in [doi:10.1137/141002293] have been
    implemented without the transform update step.

    Args:
        recon (Reconstruction): Reconstruction object containing the reconstruction
            parameters
        sources (ndarray): 4d array of sources
        measurements (ndarray): 4d array of measurements
        psfs (PSFs): PSFs object containing psfs and related data
    """
    [k,p,aa,bb] = psfs.selected_psfs.shape
    mse_inner = np.zeros((p,recon.maxiter))
    # set the initialization for the reconstruction
    if recon.recon_init_method is 'zeros':
        recon.reconstructed = np.zeros((p,1,aa,bb))

    elif recon.recon_init_method is 'tikhonov':
        recon_tik = Reconstruction(
            sources=sources,
            measurements=np.expand_dims(measurements, axis=0),
            psfs=psfs,
            deconvolver=tikhonov,
            **kwargs
        )
        recon.reconstructed = recon_tik.reconstructed
    # pre-compute some arrays for efficiency
    psfdfts_h_meas = block_mul(
        psfs.selected_psf_dfts_h,
        np.fft.fft2(measurements)
    )

    for iter in range(recon.maxiter):

        # ----- Sparse Coding -----
        patches = patch_extractor(
            recon.reconstructed,
            patch_shape=recon.patch_shape
        )

        sparse_codes = sparse_coding(
            recon,
            patches=patches,
            transform=recon.transform
        )

        # ----- Image Update -----
        Fc = np.fft.fft2(
            patch_aggregator(
                recon.transform.conj().T @ sparse_codes,
                patch_shape=recon.patch_shape,
                image_shape=(p,1,aa,bb)
            )
        )

        recon.reconstructed = np.real(
            np.fft.ifft2(
                block_mul(
                    recon.SIG_inv,
                    Fc + recon.nu * psfdfts_h_meas
                )
            )
        )

        mse_inner[:,iter] = np.mean(
                (sources - recon.reconstructed)**2,
                axis=(1, 2, 3)
        )

        dfid = recon.nu*(1/np.size(measurements))*np.sum(abs(
            block_mul(
            psfs.selected_psf_dfts, np.fft.fft2(recon.reconstructed)
        ) - np.fft.fft2(measurements)
        )**2)

        sp_error = np.sum((recon.transform@patch_extractor(
            recon.reconstructed,
            patch_shape=recon.patch_shape
        )-sparse_codes)**2)


        if iter % 20 == 0:
            print(dfid,sp_error,dfid+sp_error)
            ssim1 = np.zeros(p)
            mse1 = np.mean((sources - recon.reconstructed)**2, axis=(1, 2, 3))
            psnr1 = 20 * np.log10(np.max(sources, axis=(1,2,3))/np.sqrt(mse1))
            for i in range(p):
                ssim1[i] = ssim(sources[i,0], recon.reconstructed[i,0],
                    data_range=np.max(recon.reconstructed[i,0])-np.min(recon.reconstructed[i,0]))
            plotter4d(recon.reconstructed,
                cmap='gist_heat',
                fignum=3,
                title='Iteration: {}\n Recon. SSIM={}\n Recon. PSNR={}'.format(iter, ssim1, psnr1)
            )
            plt.pause(0.5)

    recon.mse_inner = mse_inner


# @profile
def strollr(
        recon,
        *,
        sources,
        psfs,
        measurements,
        **kwargs
):
    """Function that implements PSSI deconvolution using both sparsifying transform
    learning and emposing low rankness on the grouped patches of the images.

    The algorithm details can be found on [arXiv:1808.01316]. For efficiency,
    multiprocessing has been used in some parts of the algorithm.

    Args:
        recon (Reconstruction): Reconstruction object containing the reconstruction
            parameters
        sources (ndarray): 4d array of sources
        measurements (ndarray): 4d array of measurements
        psfs (PSFs): PSFs object containing psfs and related data
    """
    [k,p,aa,bb] = psfs.selected_psfs.shape
    mse_inner = np.zeros((p,recon.maxiter))
    psize = np.size(np.empty(kwargs['patch_shape']))

    patcher_spectrum = np.einsum(
        'ij,kl->ijkl',
        np.eye(p),
        psize * np.ones((aa,bb))
    )
    recon.SIG_inv = block_inv(
        psfs.selected_GAM +
        (recon.s + recon.lr) * patcher_spectrum
    )
    # set the initialization for the reconstruction
    if recon.recon_init_method is 'zeros':
        # recon.reconstructed = np.zeros((p,1,aa,bb))
        recon.reconstructed = 0.01*np.random.rand(p,1,aa,bb)

    elif recon.recon_init_method is 'tikhonov':
        recon_tik = Reconstruction(
            sources=sources,
            measurements=np.expand_dims(measurements, axis=0),
            psfs=psfs,
            deconvolver=tikhonov,
            **kwargs
        )
        recon.reconstructed = recon_tik.reconstructed
    # pre-compute some arrays for efficiency
    psfdfts_h_meas = block_mul(
        psfs.selected_psf_dfts_h,
        np.fft.fft2(measurements)
    )

    for iter in range(recon.maxiter):

        # ----- Low-Rank Approximation -----
        patches = patch_extractor(
            recon.reconstructed,
            patch_shape=recon.patch_shape
        )

        patch_means = np.mean(patches, axis=0)
        patches_zeromean = (patches - patch_means).T
        indices = np.zeros((patches.shape[1], recon.M), dtype=np.int)

        pool = multiprocessing.Pool()
        lowrank_i = functools.partial(lowrank,
            patches_zeromean=patches_zeromean,
            window_size=recon.window_size,
            imsize=(aa,bb),
            threshold=recon.theta,
            group_size=recon.M
        )
        D, indices = zip(*pool.map(lowrank_i, np.arange(patches.shape[1])))
        D = np.array(D)
        indices = np.array(indices, dtype=np.int)
        pool.close()
        pool.join()

        D += np.einsum('ik,j->ijk', patch_means[indices], np.ones(patches.shape[0]))

        if recon.s > 0:
            # ----- Sparse Coding -----
            patches_3d = np.zeros((patches.shape[0]*recon.l, patches.shape[1]))

            for i in range(recon.l):
                patches_3d[i*patches.shape[0]:(i+1)*patches.shape[0], :] = patches[
                    :, indices[:, i]
                ]

            sparse_codes = recon.transform @ patches_3d
            #FIXME - implement variable threshold for each image
            # sparse_indices = (sparse_codes > recon.lam).astype(np.int)
            # if not sparse_indices.any():
            #     sparse_codes = np.zeros_like(sparse_codes)
            #
            # sparse_codes = sparse_codes * sparse_indices

            for i in range(p):
                ind = np.arange(i*aa*bb, (i+1)*aa*bb)
                sparse_codes[:,ind] = hard_thresholding(
                    sparse_codes[:,ind],
                    threshold = np.sqrt(2*recon.lam[i] / recon.s)
                )

            if kwargs['learning'] is True:
                u, s, v_T = np.linalg.svd(sparse_codes @ patches_3d.T)
                recon.transform = u @ v_T

            # ----- Image Update -----
            Whb1 = np.zeros_like(patches)
            Whb = recon.transform.T @ sparse_codes
            for i in range(recon.l):
                Whb1 = indsum(Whb1, Whb[i*patches.shape[0]:(i+1)*patches.shape[0], :], indices[:, i])

            indvals = np.array(list(Counter(indices[:, :recon.l].flatten()).values()))
            indkeys = np.argsort(np.array(list(Counter(indices[:, :recon.l].flatten()).keys())))
            Whb1 = Whb1 / indvals[indkeys]

            Fc = np.fft.fft2(
                patch_aggregator(
                    Whb1,
                    patch_shape=recon.patch_shape,
                    image_shape=(p,1,aa,bb)
                )
            )
        else:
            Fc = np.zeros_like(psfdfts_h_meas)

        if recon.lr > 0:
            VhD = np.zeros_like(patches)
            for i in range(recon.M):
                VhD = indsum(VhD, D[:,:,i].T, indices[:, i])

            indvals = np.array(list(Counter(indices.flatten()).values()))
            indkeys = np.argsort(np.array(list(Counter(indices.flatten()).keys())))
            VhD = VhD / indvals[indkeys]

            Fd = np.fft.fft2(
                patch_aggregator(
                    VhD,
                    patch_shape=recon.patch_shape,
                    image_shape=(p,1,aa,bb)
                )
            )
        else:
            Fd = np.zeros_like(psfdfts_h_meas)


        recon.reconstructed = np.real(
            np.fft.ifft2(
                block_mul(
                    recon.SIG_inv,
                    recon.s * Fc + recon.lr * Fd + psfdfts_h_meas
                )
            )
        )

        if iter % 1 == 0 or iter == recon.maxiter - 1:
        # if iter == recon.maxiter - 1:
            ssim1 = np.zeros(p)
            mse1 = np.mean((sources - recon.reconstructed)**2, axis=(1, 2, 3))
            psnr1 = 20 * np.log10(np.max(sources, axis=(1,2,3))/np.sqrt(mse1))
            for i in range(p):
                ssim1[i] = ssim(sources[i,0], recon.reconstructed[i,0],
                    data_range=np.max(recon.reconstructed[i,0])-np.min(recon.reconstructed[i,0]))
            plotter4d(recon.reconstructed,
                cmap='gist_heat',
                fignum=3,
                title='Iteration: {}\n Recon. SSIM={}\n Recon. PSNR={}'.format(iter, ssim1, psnr1)
            )
            plt.pause(0.5)


def admm(
        recon,
        *,
        sources,
        psfs,
        measurements,
        **kwargs
):
    """Function that implements PSSI deconvolution with the ADMM algorithm using
    total variation or patch based sparsifying transform optionally.

    The cost function has the form:
    x_hat = argmin_{x} { ||Ax-y||_2^2 + \lambda ||Wx||_1 }
    for TV, W is the linear gradient operator acting on the individual spectral
    images x_i. For patch based transform, W acts on patches of x_i's. So W has
    the form W=QP where P has patch extractors, and Q has the sparsifying transforms
    in their blocks.

    Args:
        recon (Reconstruction): Reconstruction object containing the reconstruction
            parameters
        sources (ndarray): 4d array of sources
        measurements (ndarray): 4d array of measurements
        psfs (PSFs): PSFs object containing psfs and related data
    """

    [k,p,aa,bb] = psfs.selected_psfs.shape
    mse_inner = np.zeros((p,recon.maxiter))
    cost = np.zeros(recon.maxiter+1)
    itererror = np.zeros(recon.maxiter)
    cost[0] = 0.5 * np.sum(measurements**2)
    #initialize the reconstruction
    if recon.recon_init_method is 'zeros':
        primal1 = np.zeros((p,1,aa,bb))

    elif recon.recon_init_method is 'tikhonov':
        recon_tik = Reconstruction(
            sources=sources,
            measurements=np.expand_dims(measurements, axis=0),
            psfs=psfs,
            deconvolver=tikhonov,
            **kwargs
        )
        primal1 = recon_tik.reconstructed

    primal2 = domain_transformer(primal1, recon, **kwargs)
    dual = primal2.copy()
    # pre-compute some arrays for efficiency
    psfdfts_h_meas = block_mul(
        psfs.selected_psf_dfts_h,
        np.fft.fft2(measurements)
    )

    for iter in range(recon.maxiter):

        primal1_old = primal1.copy()
        ###### primal1 update ######
        primal1 = np.real(
            np.fft.ifft2(
                block_mul(
                    recon.SIG_inv,
                    psfdfts_h_meas +
                    recon.nu * np.fft.fft2(
                        domain_transformer_T(primal2-dual, recon, **kwargs, shape=sources.shape)
                    )
                )
            )
        )
        ###### primal2 update ######
        pre_primal2 = domain_transformer(primal1, recon, **kwargs)

        #FIXME - implement variable threshold for each image
        # primal2 = thresholding(
        #     pre_primal2 + dual, recon, **kwargs
        # )

        for i in range(p):
            ind = np.arange(i*aa*bb, (i+1)*aa*bb)
            primal2[:,ind] = hard_thresholding(
                pre_primal2[:,ind] + dual[:,ind],
                threshold = np.sqrt(2*recon.lam[i] / recon.nu)
            )

        ##### dual update #####
        dual += (pre_primal2 - primal2)

        # if kwargs['learning'] is True:
        #     u,s,vT = np.linalg.svd((recon.nu*primal2-dual) @ recon.patches.T)
        #     recon.transform = u @ vT

        if kwargs['learning'] is True:
            u,s,vT = np.linalg.svd((primal2-dual) @ recon.patches.T)
            recon.transform = u @ vT

        # mse_inner[:,iter] = np.mean(
        #         (sources - primal1)**2,
        #         axis=(1, 2, 3)
        # )
        #
        # dfid = 0.5*(1/np.size(measurements))*np.sum(abs(
        #     block_mul(
        #     psfs.selected_psf_dfts, np.fft.fft2(primal1)
        # ) - np.fft.fft2(measurements)
        # )**2)

        # l1_reg = recon.lam * np.sum(abs(primal2))
        # l1_reg2 = recon.lam * np.sum(abs(pre_primal2))
        #
        # residual = 0.5*recon.nu * np.sum((pre_primal2 - primal2)**2)
        #
        # lagrange = recon.nu * np.sum(dual * (pre_primal2 - primal2))

        # print(dfid, l1_reg, lagrange, residual, dfid+l1_reg+residual+lagrange)
        # cost[iter+1] = dfid + l1_reg2
        # itererror[iter] = np.sum((primal1-primal1_old)**2) / np.sum(primal1**2)

        if iter % 100 == 0 or iter == recon.maxiter - 1:
            # print(dfid, l1_reg2, dfid+l1_reg2)
            ssim1 = np.zeros(p)
            mse1 = np.mean((sources - primal1)**2, axis=(1, 2, 3))
            psnr1 = 20 * np.log10(np.max(sources, axis=(1,2,3))/np.sqrt(mse1))
            for i in range(p):
                ssim1[i] = ssim(sources[i,0], primal1[i,0],
                    data_range=np.max(primal1[i,0])-np.min(primal1[i,0]))
            plotter4d(primal1,
                cmap='gist_heat',
                fignum=3,
                title='Iteration: {}\n Recon. SSIM={}\n Recon. PSNR={}'.format(iter, ssim1, psnr1)
            )
            plt.pause(0.5)

    # recon.mse_inner = mse_inner
    # recon.dfid.append(2*dfid)
    # recon.reg.append(l1_reg2/recon.lam)
    # recon.cost = cost
    # recon.itererror = itererror
    recon.reconstructed = primal1

def tikhonov(recon, *, psfs, measurements, **kwargs):
    """Perform Tikhonov regularization based image reconstruction for PSSI.

    Args:
        psfs (PSFs): PSFs object containing psfs and other csbs state data
        measured_noisy (ndarray): 4d array of noisy measurements

    Returns:
        4d array of reconstructed images
    """
    recon.reconstructed = np.real(
        np.fft.ifft2(
                block_mul(
                    recon.SIG_e_dft_inv,
                    block_mul(
                        psfs.selected_psf_dfts_h,
                        np.fft.fft2(measurements)
                    )
                )
        )
    )

    recon.dfid.append((1/np.size(measurements))*np.sum(abs(
            block_mul(
            psfs.selected_psf_dfts, np.fft.fft2(recon.reconstructed)
        ) - np.fft.fft2(measurements)
        )**2))

    lamo = np.einsum('ij,kl', np.eye(psfs.selected_psfs.shape[1]), recon.LAM)
    recon.reg.append(1/np.size(recon.reconstructed)*np.sum(block_mul(lamo, abs(np.fft.fft2(recon.reconstructed))**2)))
    #FIXME
    # for i in range(recon.reconstructed.shape[0]):
    #     recon.reconstructed[i] -= np.mean(recon.reconstructed[i])


def pre_computer(
    recon,
    *,
    psfs,
    measurements,
    deconvolver,
    **kwargs
):
    """ Pre compute the required arrays that will remain unchanged through the
    algorithm iterations, for efficiency.

    Args:
        recon (Reconstruction): Reconstruction object containing the reconstruction
            parameters
        measurements (ndarray): 4d array of measurements
        deconvolver (function): user specified function for the deconvolution
        psfs (PSFs): PSFs object containing psfs and related data
    """

    recon.pre_computed = {}

    if deconvolver is tikhonov:
        [_,num_sources,aa,bb] = psfs.selected_psfs.shape
        if kwargs['tikhonov_scale'] is 'full':
            if kwargs['tikhonov_matrix'] is 'derivative':
                recon.LAM = get_LAM(rows=aa,cols=bb,order=kwargs['tikhonov_order'])
            elif kwargs['tikhonov_matrix'] is 'covariance':
                b = np.load('/home/kamo/tmp/covariance_learning/cov_impulse5.npy')
                # b = np.fft.fftshift(b)
                recon.LAM = 1 / np.fft.fft2(b)
        elif kwargs['tikhonov_scale'] is 'patch':
            if kwargs['tikhonov_matrix'] is 'derivative':
                [pa,pb,_] = kwargs['patch_shape']
                dif = np.zeros((pa,pb))
                dif[0,0] = 1
                dif[1:,:-1] -= np.eye(pa-1)
                dif[1:,1:] += np.eye(pa-1)
                difx = np.kron(np.eye(pa), dif)
                dify = np.kron(dif, np.eye(pa))
                covdif = difx.T @ difx + dify.T @ dify
                im0 = np.zeros((1,1,aa,bb))
                im0[0,0,0,0] = 1
                G1 = patch_aggregator(
                    covdif @ patch_extractor(
                    im0, patch_shape=kwargs['patch_shape']),
                    patch_shape=kwargs['patch_shape'],
                    image_shape=(1,1,aa,bb)
                )[0,0]
                recon.LAM = np.fft.fft2(G1)
            elif kwargs['tikhonov_matrix'] is 'covariance':
                cov = np.load('/home/kamo/tmp/covariance_learning/cov_patch2.npy')
                cov = np.random.random((36,36))
                im0 = np.zeros((1,1,aa,bb))
                im0[0,0,0,0] = 1
                G1 = patch_aggregator(
                    np.linalg.inv(cov) @ patch_extractor(
                    im0, patch_shape=kwargs['patch_shape']),
                    patch_shape=kwargs['patch_shape'],
                    image_shape=(1,1,aa,bb)
                )[0,0]
                recon.LAM = np.fft.fft2(G1)

        recon.pre_computed['tikhonov_lam'] = {'SIG_e_dft_inv':[]}
        for lam in kwargs['tikhonov_lam']:
            recon.pre_computed['tikhonov_lam']['SIG_e_dft_inv'].append([lam,
                block_inv(
                    psfs.selected_GAM +
                    lam * np.einsum('ij,kl', np.eye(num_sources), recon.LAM)
                )]
            )

    elif deconvolver is admm_tv:
        [_,num_sources,aa,bb] = psfs.selected_psfs.shape
        recon.LAM = get_LAM(rows=aa,cols=bb,order=1)

        recon.pre_computed['nu'] = {'SIG_inv':[]}
        for nu in kwargs['nu']:
            recon.pre_computed['nu']['SIG_inv'].append([nu,
                block_inv(
                    psfs.selected_GAM +
                    #FIXME
                    nu * np.einsum('ij,kl', np.eye(num_sources), recon.LAM)
                    # nu * np.einsum('ij,kl', np.eye(num_sources), np.ones((aa,bb)))
                )]
            )

    elif deconvolver is fista_dct:
        [_,num_sources,aa,bb] = psfs.selected_psfs.shape
        diffx_kernel = np.zeros((aa,bb))
        diffy_kernel = np.zeros((aa,bb))
        diffx_kernel[0,0] = 1 ; diffx_kernel[0,1] = -1
        diffy_kernel[0,0] = 1 ; diffy_kernel[1,0] = -1
        lamx = np.fft.fft2(diffx_kernel)
        lamy = np.fft.fft2(diffy_kernel)
        lamx[abs(lamx)==0] = 1e10
        lamy[abs(lamy)==0] = 1e10

        lamx = np.einsum('ij,kl', np.eye(num_sources), lamx)
        lamy = np.einsum('ij,kl', np.eye(num_sources), lamy)

        recon.LAM = np.concatenate((lamx, lamy), axis=0)
        recon.LAMinv = np.concatenate((0.5/lamx, 0.5/lamy), axis=1)

        recon.SIG = block_mul(
            block_herm(recon.LAMinv), block_mul(psfs.selected_GAM, recon.LAMinv)
        )


    elif deconvolver is sparsepatch:
        [_,num_sources,aa,bb] = psfs.selected_psfs.shape
        psize = np.size(np.empty(kwargs['patch_shape']))

        # FIXME: this assumes that the transform is unitary
        patcher_spectrum = np.einsum(
            'ij,kl->ijkl',
            np.eye(num_sources),
            psize * np.ones((aa,bb))
        )
        recon.pre_computed['nu'] = {'SIG_inv':[]}
        for nu in kwargs['nu']:
            recon.pre_computed['nu']['SIG_inv'].append([nu,
                block_inv(
                    nu * psfs.selected_GAM +
                    patcher_spectrum
                )]
            )

    elif deconvolver is strollr:
        pass

    elif deconvolver is admm:
        [_,num_sources,aa,bb] = psfs.selected_psfs.shape
        psize = np.size(np.empty(kwargs['patch_shape']))
        if kwargs['sparsifyer'] is 'TV':
            recon.LAM = get_LAM(rows=aa,cols=bb,order=1)
        elif kwargs['sparsifyer'] is 'patch_based':
            recon.LAM = psize * np.ones((aa,bb))

        spectrum = np.einsum(
            'ij,kl->ijkl',
            np.eye(num_sources),
            recon.LAM
        )
        recon.GAM = block_mul(
            psfs.selected_psf_dfts_h,
            psfs.selected_psf_dfts
        )
        recon.pre_computed['nu'] = {'SIG_inv':[]}
        for nu in kwargs['nu']:
            recon.pre_computed['nu']['SIG_inv'].append([nu,
                block_inv(
                    recon.GAM +
                    nu * spectrum
                )]
            )

class Reconstruction():
    """A class for holding the reconstructions and related parameters

    Args:
        sources (ndarray): 4d array of sources
        measurements (ndarray): 5d array of measurements
        psfs (PSFs): PSFs object containing psfs and related data
        deconvolver (def): function to perform deconvolution (default, mas.pssi_deconvolution.tikhonov)

    Attributes:
        dfid (ndarray): 1d array of data fidelity terms for a performed deconvolution
            its length is equal to the number of iterations
        reg (ndarray): 1d array of regularization terms for a performed deconvolution
            its length is equal to the number of iterations
        mse (ndarray): array of reconstruction MSEs for performed deconvolution(s)
            its shape depends on the number of noise realizations and the
            iterable parameters of the specified deconvolution algorithm
        mse_average (ndarray): the same with mse, except that it is averaged
            over the different noise realizations
        psnr (ndarray): PSNR counterpart of the mse array
        psnr_average (ndarray): PSNR counterpart of the mse_average array
        reconstructed (ndarray): array containing the last reconstructed images.
            its shape is equal to that of the `sources`

        learning (bool) [optional]: Bool specifying if transform learning is On or Off
        tv (str) [optional]: ['iso' or 'aniso'] string specifying which type
            of TV norm is used in the fista_tv algorithm
        sparsity_ratio_array (list) [optional]: list of sparsity ratios used
            in the sparsepatch algorithm
        sparsity_threshold_array (list) [optional]: list of sparsity thresholds
            used in the sparsepatch algorithm
        reconstructed_array (ndarray) [optional]: array containing all of the
            `reconstructed` arrays for every parameter and noise realization
            combination.
        recon_init_method (str) [optional]: ['zeros' or 'tikhonov'] string
            specifying the initialization method for the iterative algorithms.
        maxiter (int) [optional]: maximum number of iterations.
        patch_shape (tuple) [optional]: tuple specifying the patch shape.
        transform (ndarray) [optional]: patch based sparsifying transform at the
            final iteration
        nu_array (list) [optional]: list of `nu` parameters used by the deconvolution
            algorithm
        lam_array (list) [optional]: list of `lam` parameters used by the deconvolution
            algorithm
        tikhonov_lam (float) [optional]: regularization parameter of tikhonov
        tikhonov_order (int) [optional]: [0,1 or 2] order of the gradient
            operator used in the tikhonov regularization
        M (int) [optional]: for the strollr algorithm, its the number of patches
            in a group on which low rankness is imposed
        l (int) [optional]: for the strolr algorithm, its the number of patches
            in a group on which 3D sparsifying transform is applied
        window_size (tuple) [optional]: for the strolr algorithm, its the window
            size on which similar patches are searched for with respect to the
            patch in the middle
        s_array (list) [optional]: for strollr, its the list of sparsity parameters - `s`
        lr_array (list) [optional]: for strollr, its the list of low-rankness
            parameters - `lr`
        theta (list) [optional]: for strollr, its the list of low-rankness
            parameters - `theta`. Its length is equal to the number of sources.
    """

    def __init__(
        self,
        *,
        psfs,
        sources,
        measurements,
        deconvolver,
        **kwargs
    ):

        iterable_params = {}
        num_instances = measurements.shape[0]

        if deconvolver is tikhonov:
            if isinstance(kwargs['tikhonov_lam'], (int, float)):
                kwargs['tikhonov_lam'] = [kwargs['tikhonov_lam']]
            iterable_params['tikhonov_lam'] = kwargs['tikhonov_lam']

            #set the parameters as the attributes of the Reconstruction object
            self.tikhonov_lam_array = kwargs['tikhonov_lam']
            self.order = kwargs['tikhonov_order']

        elif deconvolver is admm:
            if isinstance(kwargs['nu'], (int, float)):
                kwargs['nu'] = [kwargs['nu']]
            iterable_params['nu'] = kwargs['nu']
            if isinstance(kwargs['lam'], (int, float)):
                kwargs['lam'] = [kwargs['lam']]
            #FIXME - this is done for variable threshold for each image
            # iterable_params['lam'] = kwargs['lam']

            #set the parameters as the attributes of the Reconstruction object
            self.recon_init_method = kwargs['recon_init_method']
            self.maxiter = kwargs['maxiter']
            self.patch_shape = kwargs['patch_shape']
            self.transform = kwargs['transform']
            self.nu_array = kwargs['nu']
            #FIXME - this is done for variable threshold for each image
            self.lam = kwargs['lam']
            if 'tikhonov_lam' in kwargs:
                self.tikhonov_lam = kwargs['tikhonov_lam']
                self.tikhonov_order = kwargs['tikhonov_order']

        elif deconvolver is strollr:
            if isinstance(kwargs['s'], (int, float)):
                kwargs['s'] = [kwargs['s']]
            iterable_params['s'] = kwargs['s']
            if isinstance(kwargs['lr'], (int, float)):
                kwargs['lr'] = [kwargs['lr']]
            iterable_params['lr'] = kwargs['lr']
            if isinstance(kwargs['lam'], (int, float)):
                kwargs['lam'] = [kwargs['lam']]
            #FIXME - this is done for variable threshold for each image
            # iterable_params['lam'] = kwargs['lam']
            # if isinstance(kwargs['theta'], (int, float)):
            #     kwargs['theta'] = [kwargs['theta']]
            # iterable_params['theta'] = kwargs['theta']

            #set the parameters as the attributes of the Reconstruction object
            self.recon_init_method = kwargs['recon_init_method']
            self.window_size = kwargs['window_size']
            self.maxiter = kwargs['maxiter']
            self.patch_shape = kwargs['patch_shape']
            self.transform = kwargs['transform']
            self.M = kwargs['M']
            self.l = kwargs['l']
            self.s_array = kwargs['s']
            self.lr_array = kwargs['lr']
            self.theta = kwargs['theta']
            #FIXME - this is done for variable threshold for each image
            self.lam = kwargs['lam']
            if 'tikhonov_lam' in kwargs:
                self.tikhonov_lam = kwargs['tikhonov_lam']
                self.tikhonov_order = kwargs['tikhonov_order']

        elif deconvolver is sparsepatch:
            if isinstance(kwargs['nu'], (int, float)):
                kwargs['nu'] = [kwargs['nu']]
            iterable_params['nu'] = kwargs['nu']

            #set the parameters as the attributes of the Reconstruction object
            self.recon_init_method = kwargs['recon_init_method']
            self.maxiter = kwargs['maxiter']
            self.patch_shape = kwargs['patch_shape']
            self.transform = kwargs['transform']
            self.learning = kwargs['learning']
            self.nu_array = kwargs['nu']
            if 'tikhonov_lam' in kwargs:
                self.tikhonov_lam = kwargs['tikhonov_lam']
                self.tikhonov_order = kwargs['tikhonov_order']
            if kwargs['sparsity_ratio'] is not None:
                if isinstance(kwargs['sparsity_ratio'], (int, float)):
                    kwargs['sparsity_ratio'] = [kwargs['sparsity_ratio']]
                self.sparsity_ratio_array = kwargs['sparsity_ratio']
                iterable_params['sparsity_ratio'] = kwargs['sparsity_ratio']

            else:
                if isinstance(kwargs['sparsity_threshold'], (int, float)):
                    kwargs['sparsity_threshold'] = [kwargs['sparsity_threshold']]
                self.sparsity_threshold_array = kwargs['sparsity_threshold']
                iterable_params['sparsity_threshold'] = kwargs['sparsity_threshold']

        elif deconvolver is fista_dct:
            if isinstance(kwargs['nu'], (int, float)):
                kwargs['nu'] = [kwargs['nu']]
            iterable_params['nu'] = kwargs['nu']

            if isinstance(kwargs['lam'], (int, float)):
                kwargs['lam'] = [kwargs['lam']]
            iterable_params['lam'] = kwargs['lam']

            #set the parameters as the attributes of the Reconstruction object
            self.recon_init_method = kwargs['recon_init_method']
            self.maxiter = kwargs['maxiter']
            self.nu_array = kwargs['nu']
            self.lam_array = kwargs['lam']
            if 'tikhonov_lam' in kwargs:
                self.tikhonov_lam = kwargs['tikhonov_lam']
                self.tikhonov_order = kwargs['tikhonov_order']

        elif deconvolver is fista_tv:
            if isinstance(kwargs['nu'], (int, float)):
                kwargs['nu'] = [kwargs['nu']]
            iterable_params['nu'] = kwargs['nu']

            if isinstance(kwargs['lam'], (int, float)):
                kwargs['lam'] = [kwargs['lam']]
            iterable_params['lam'] = kwargs['lam']

            #set the parameters as the attributes of the Reconstruction object
            self.tv = kwargs['tv']
            self.recon_init_method = kwargs['recon_init_method']
            self.maxiter = kwargs['maxiter']
            self.nu_array = kwargs['nu']
            self.lam_array = kwargs['lam']
            if 'tikhonov_lam' in kwargs:
                self.tikhonov_lam = kwargs['tikhonov_lam']
                self.tikhonov_order = kwargs['tikhonov_order']

        elif deconvolver is admm_tv:
            if isinstance(kwargs['nu'], (int, float)):
                kwargs['nu'] = [kwargs['nu']]
            iterable_params['nu'] = kwargs['nu']

            if isinstance(kwargs['lam'], (int, float)):
                kwargs['lam'] = [kwargs['lam']]
            iterable_params['lam'] = kwargs['lam']

            #set the parameters as the attributes of the Reconstruction object
            self.recon_init_method = kwargs['recon_init_method']
            self.maxiter = kwargs['maxiter']
            self.nu_array = kwargs['nu']
            self.lam_array = kwargs['lam']
            if 'tikhonov_lam' in kwargs:
                self.tikhonov_lam = kwargs['tikhonov_lam']
                self.tikhonov_order = kwargs['tikhonov_order']

        pre_computer(
            self,
            psfs=psfs,
            measurements=measurements,
            deconvolver=deconvolver,
            **kwargs
        )

        # ----- run the deconvolution algorithm for each noise realization -----
        if kwargs['iterproduct'] is True:
            iterparams = itertools.product(*iterable_params.values())
            dim = (num_instances,) + tuple([len(i) for i in iterable_params.values()])
        else:
            iterparams = np.array([i for i in iterable_params.values()]).T
            dim = (num_instances,) + (iterparams.shape[0],)
        if kwargs['store_recons'] is True:
            self.reconstructed_array = np.zeros((dim+sources.shape))
        self.mse = np.zeros(dim+(sources.shape[0],))
        self.psnr = np.zeros(dim+(sources.shape[0],))
        self.ssim = np.zeros(dim+(sources.shape[0],))
        if hasattr(self, 'maxiter'):
            self.mse_inner_array = np.zeros(dim+(sources.shape[0],self.maxiter))
            self.cost_array = np.zeros(dim+(self.maxiter+1,))
            self.itererror_array = np.zeros(dim+(self.maxiter,))

        self.dfid = []
        self.reg = []
        if hasattr(self, 'maxiter'):
            self.mse_inner = np.zeros(dim+(sources.shape[0], self.maxiter))

        for params in iterparams:
            params_dict = dict(zip(iterable_params.keys(),params))

            for param in params_dict:
                setattr(self, param, params_dict[param])
                if param in self.pre_computed:
                    for array in self.pre_computed[param]:
                        for i in self.pre_computed[param][array]:
                            if i[0] == params_dict[param]:
                                setattr(self, array, i[1])

            for instance in range(num_instances):
                ind = []
                for param in params_dict:
                    ind.append(np.where(
                        np.array(iterable_params[param])==np.array(params_dict[param])
                    )[0][0])
                ind = (instance,) + tuple(ind)
                if kwargs['iterproduct'] is False:
                    ind = ind[:2]
                kwargs['ind'] = ind

                logging.info('{}/{}'.format(tuple([k+1 for k in ind]), dim))

                deconvolver(
                    self,
                    measurements=measurements[instance],
                    sources=sources,
                    psfs=psfs,
                    **kwargs
                )
                if kwargs['store_recons'] is True:
                    self.reconstructed_array[ind]=self.reconstructed
                self.mse[ind] = np.mean((sources - self.reconstructed)**2, axis=(1, 2, 3))
                self.psnr[ind] = 20 * np.log10(np.max(sources, axis=(1,2,3))/np.sqrt(self.mse[ind]))
                for i in range(sources.shape[0]):
                    self.ssim[ind+(i,)] = ssim(sources[i,0], self.reconstructed[i,0],
                        data_range=self.reconstructed[i,0].max()-self.reconstructed[i,0].min())
                if hasattr(self, 'mse_inner'):
                    self.mse_inner_array[ind] = self.mse_inner
        self.mse_average = np.mean(self.mse, axis=0)
        self.psnr_average = np.mean(self.psnr, axis=0)
        self.ssim_average = np.mean(self.ssim, axis=0)
        self.dfid = np.array(self.dfid)
        self.reg = np.array(self.reg)

        if kwargs['lcurve'] is True:
            param = list(kwargs['lcurve_param'].values())[0]
            paramlog = np.log10(param)
            dfidlog = np.log10(self.dfid)

            for i in np.arange(len(dfidlog)-1):
                if dfidlog[i] == dfidlog[i+1]:
                    dfidlog[i] -= 1e-3*dfidlog[i]

            reglog = np.log10(self.reg)
            for i in range(len(dfidlog)):
                dfidlog_fine = np.linspace(dfidlog[0], dfidlog[-1], num=100, endpoint=True)
                curve = interp1d(dfidlog, reglog, kind='cubic')
                reglog_fine = curve(dfidlog_fine)
                if reglog_fine.max() <= reglog.max() and reglog_fine.min() >= reglog.min():
                    break
                else:
                    dfidlog = dfidlog[:-1]
                    reglog = reglog[:-1]
                    paramlog = paramlog[:-1]

            paramlog_fine = np.linspace(paramlog[0], paramlog[-1], num=100, endpoint=True)
            dfid2param = interp1d(dfidlog, paramlog, kind='cubic')
            param2dfid = interp1d(paramlog, dfidlog, kind='cubic')
            param2reg = interp1d(paramlog, reglog, kind='cubic')
            curvature = np.convolve(reglog_fine, [1,-2,1], 'valid')
            dlogfine = param2dfid(paramlog_fine)
            rlogfine = param2reg(paramlog_fine)
            curvd = np.convolve(dlogfine, [1,-2,1], 'valid')
            curvr = np.convolve(rlogfine, [1,-2,1], 'valid')
            curv = curvr/curvd

            for i in range(len(curvature)):
                idx_max = np.argmax(curvature[i:])
                if idx_max > 1:
                    idx_max += i
                    break

            opt_param = np.power(10, dfid2param(dfidlog_fine[idx_max + 2]))
            kwargs['lcurve'] = False
            kwargs[list(kwargs['lcurve_param'].keys())[0]] = opt_param
            self.lcurverecon = Reconstruction(
                sources=sources,
                measurements=measurements,
                psfs=psfs,
                deconvolver=deconvolver,
                **kwargs,
            )

        # if kwargs['lcurve'] is True:
        #     param = list(kwargs['lcurve_param'].values())[0]
        #     paramlog = np.log10(param)
        #     dfidlog = np.log10(self.dfid)
        #     reglog = np.log10(self.reg)
        #     for i in np.arange(len(dfidlog)-1):
        #         if dfidlog[i] == dfidlog[i+1]:
        #             dfidlog[i] -= 1e-3*dfidlog[i]
        #
        #     for i in range(len(dfidlog)):
        #         paramlog_fine = np.linspace(paramlog[0], paramlog[-1], num=100, endpoint=True)
        #         curve = interp1d(dfidlog, reglog, kind='cubic')
        #         reglog_fine = curve(dfidlog_fine)
        #         if reglog_fine.max() <= reglog.max() and reglog_fine.min() >= reglog.min():
        #             break
        #         else:
        #             dfidlog = dfidlog[:-1]
        #             reglog = reglog[:-1]
        #             paramlog = paramlog[:-1]
        #
        #     dfid2param = interp1d(dfidlog, paramlog, kind='cubic')
        #     param2dfid = interp1d(paramlog, dfidlog, kind='cubic')
        #     curve_reg = np.convolve(reglog_fine, [1,-2,1], 'valid')
        #     curve_dfid = np.convolve(dfidlog_fine, [1,-2,1], 'valid')
        #     for i in range(len(curvature)):
        #         idx_max = np.argmax(curvature[i:])
        #         if idx_max > 1:
        #             idx_max += i
        #             break
        #
        #     opt_param = np.power(10, dfid2param(dfidlog_fine[idx_max + 2]))
        #     import ipdb; ipdb.set_trace()
        #     kwargs['lcurve'] = False
        #     kwargs[list(kwargs['lcurve_param'].keys())[0]] = opt_param
        #     kwargs['maxiter'] = 500
        #     #FIXME
        #     kwargs['nu'] = kwargs['lam']
        #     self.lcurverecon = Reconstruction(
        #         sources=sources,
        #         measurements=measurements,
        #         psfs=psfs,
        #         deconvolver=deconvolver,
        #         **kwargs,
        #     )
