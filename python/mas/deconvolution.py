#!/usr/bin/env python3
# Ulas Kamaci 2018-08-27

import numpy as np
import logging, itertools, functools, multiprocessing, pybm3d
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

def sparse_coding(recon, *, patches, threshold, transform):
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

def block_matching(i, patches_zeromean, window_size, imsize, group_size):
    """Form a group of similar patches to `i`th patch

    Since this function is computationally expensive, the similar patches are
    searched only in a window around the given patch, whose size is specified
    by `window_size`. Euclidean distance is used as the similarity metric.

    Args:
        i (int): index of the patch of interest
        patches_zeromean (ndarray): 2d array of zero mean patches
        window_size (tuple): tuple of size of the window in which the patches are searched
        imsize (tuple): length 2 tuple of size of the reconstructed image
        group_size (int): number of patches in the group to be formed

    Returns:
        ind (ndarray): array of indices of selected patches that are close to
            the `i`th patch
    """
    # get the indices inside the window
    ind_wind = ind_selector(i, imsize=imsize, window_size=window_size)

    ind = ind_wind[
        np.argsort(
            np.linalg.norm(patches_zeromean[ind_wind] - patches_zeromean[i], axis=1)
        )[:group_size]
    ]
    return patches_zeromean[ind].T, ind

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

def sparsepatch(
            *,
            sources,
            psfs,
            measurements,
            recon_init_method,
            tikhonov_lam,
            tikhonov_order,
            iternum,
            nu,
            lam,
            patch_shape,
            transform,
            learning
):
    """Function that implements PSSI deconvolution with a patch based sparsifying
    transform for sparse recovery.

    P1 and P3 formulations described in [doi:10.1137/141002293] have been
    implemented without the transform update step.


    Args:
        sources (ndarray): 4d array of sources
        psfs (PSFs): PSFs object containing psfs and related data
        measurements (ndarray): 4d array of measurements
        recon_init_method (str): ['zeros' or 'tikhonov'] string specifying the
            initialization method for the reconstructions
        tikhonov_lam (float): regularization parameter of tikhonov
        tikhonov_order (int): [0,1 or 2] order of the discrete derivative
            operator used in the tikhonov regularization
        iternum (int): number of iterations of ADMM
        nu (float): augmented Lagrangian parameter
        lam (float): penalty parameter of the sparsity term
        patch_shape (tuple): tuple of the shape of the patches used
        transform (ndarray): ndarray of the sparsifying transform used
        learning (bool): boolean variable of whether the transform gets updated
    """
    [k,num_sources,aa,bb] = psfs.selected_psfs.shape[:2] + sources.shape[2:]
    psize = np.size(np.empty(patch_shape))
    # mse_inner = np.zeros((num_sources,recon.maxiter))
    if type(lam) is np.float or type(lam) is np.int:
        lam = np.ones(num_sources) * lam

    ################## initialize the primal/dual variables ##################
    if recon_init_method is 'zeros':
        recon = np.zeros((num_sources,1,aa,bb))

    elif recon_init_method is 'tikhonov':
        recon = tikhonov(
            sources=sources,
            psfs=psfs,
            measurements=measurements,
            tikhonov_lam=tikhonov_lam,
            tikhonov_order=tikhonov_order
        )

    ################# pre-compute some arrays for efficiency #################
    psfdfts_h_meas = block_mul(
        psfs.selected_psf_dfts_h,
        np.fft.fft2(measurements)
    )
    LAM = psize * np.ones((aa,bb))
    spectrum = nu * np.einsum('ij,kl->ijkl', np.eye(num_sources), LAM)
    SIG_inv = block_inv(psfs.selected_GAM + spectrum)

    for iter in range(iternum):

        # ----- Sparse Coding -----
        patches = patch_extractor(
            recon,
            patch_shape=patch_shape
        )

        sparse_codes = transform @ patches
        for i in range(num_sources):
            sparse_codes[:,i*aa*bb:(i+1)*aa*bb] = hard_thresholding(
                sparse_codes[:,i*aa*bb:(i+1)*aa*bb],
                threshold=np.sqrt(lam[i]/nu)
            )

        # ----- Image Update -----
        Fc = np.fft.fft2(
            patch_aggregator(
                transform.conj().T @ sparse_codes,
                patch_shape=patch_shape,
                image_shape=(num_sources,1,aa,bb)
            )
        )

        recon = np.real(
            np.fft.ifft2(
                block_mul(
                    SIG_inv,
                    nu*Fc + psfdfts_h_meas
                )
            )
        )

        if learning is True:
            u,s,vT = np.linalg.svd(sparse_codes @ patches.T)
            transform = u @ vT

        # mse_inner[:,iter] = np.mean(
        #         (sources - recon.reconstructed)**2,
        #         axis=(1, 2, 3)
        # )
        #
        # dfid = recon.nu*(1/np.size(measurements))*np.sum(abs(
        #     block_mul(
        #     psfs.selected_psf_dfts, np.fft.fft2(recon.reconstructed)
        # ) - np.fft.fft2(measurements)
        # )**2)
        #
        # sp_error = np.sum((recon.transform@patch_extractor(
        #     recon.reconstructed,
        #     patch_shape=recon.patch_shape
        # )-sparse_codes)**2)


        if iter % 1 == 0:
            # print(dfid,sp_error,dfid+sp_error)
            ssim1 = np.zeros(num_sources)
            mse1 = np.mean((sources - recon)**2, axis=(1, 2, 3))
            psnr1 = 20 * np.log10(np.max(sources, axis=(1,2,3))/np.sqrt(mse1))
            for i in range(num_sources):
                ssim1[i] = ssim(sources[i,0], recon[i,0],
                    data_range=np.max(recon[i,0])-np.min(recon[i,0]))
            plotter4d(recon,
                cmap='gist_heat',
                fignum=3,
                figsize=(5.6,8),
                title='Iteration: {}\n Recon. SSIM={}\n Recon. PSNR={}'.format(iter, ssim1, psnr1)
            )
            plt.pause(0.5)

    return recon

# @profile
def strollr(
        *,
        sources,
        psfs,
        measurements,
        recon_init_method,
        tikhonov_lam,
        tikhonov_order,
        iternum,
        lr,
        theta,
        s,
        lam,
        patch_shape,
        transform,
        learning,
        window_size,
        group_size,
        group_size_s
):
    """Function that implements PSSI deconvolution using both sparsifying transform
    learning and emposing low rankness on the grouped patches of the images.

    The algorithm details can be found on [arXiv:1808.01316]. For efficiency,
    multiprocessing has been used in some parts of the algorithm.

    Args:
        sources (ndarray): 4d array of sources
        psfs (PSFs): PSFs object containing psfs and related data
        measurements (ndarray): 4d array of measurements
        recon_init_method (str): ['zeros' or 'tikhonov'] string specifying the
            initialization method for the reconstructions
        tikhonov_lam (float): regularization parameter of tikhonov
        tikhonov_order (int): [0,1 or 2] order of the discrete derivative
            operator used in the tikhonov regularization
        iternum (int): number of iterations of ADMM
        lr (float): augmented Lagrangian parameter of the low-rank term
        theta (float): penalty parameter of the low-rank term
        s (float): augmented Lagrangian parameter of the sparsity term
        lam (float): penalty parameter of the sparsity term
        patch_shape (tuple): tuple of the shape of the patches used
        transform (ndarray): ndarray of the sparsifying transform used
        learning (bool): boolean variable of whether the transform gets updated
        window_size (tuple): tuple of the window size over which the group of
            similar patches to the centered reference patch is searched
        group_size (int): the number of patches in each group (of similar patches)
        group_size_s (int): the number of patches in each group (for sparsity)
    """
    [k,num_sources,aa,bb] = psfs.selected_psfs.shape[:2] + sources.shape[2:]
    psize = np.size(np.empty(patch_shape))
    if type(theta) is np.float:
        theta = np.ones(num_sources) * theta
    if type(lam) is np.float:
        lam = np.ones(num_sources) * lam

    ################## initialize the primal/dual variables ##################
    if recon_init_method is 'zeros':
        recon = np.zeros((num_sources,1,aa,bb))

    elif recon_init_method is 'tikhonov':
        recon = tikhonov(
            sources=sources,
            psfs=psfs,
            measurements=measurements,
            tikhonov_lam=tikhonov_lam,
            tikhonov_order=tikhonov_order
        )

    ################# pre-compute some arrays for efficiency #################
    patcher_spectrum = np.einsum(
        'ij,kl->ijkl',
        np.eye(num_sources),
        psize * np.ones((aa,bb))
    )
    SIG_inv = block_inv(
        psfs.selected_GAM +
        (s + lr) * patcher_spectrum
    )
    psfdfts_h_meas = block_mul(
        psfs.selected_psf_dfts_h,
        np.fft.fft2(measurements)
    )

    for iter in range(iternum):

        # ----- Low-Rank Approximation -----
        patches = patch_extractor(
            recon,
            patch_shape=patch_shape
        )

        patch_means = np.mean(patches, axis=0)
        patches_zeromean = (patches - patch_means).T
        indices = np.zeros((patches.shape[1], group_size), dtype=np.int)

        pool = multiprocessing.Pool()
        lowrank_i = functools.partial(lowrank,
            patches_zeromean=patches_zeromean,
            window_size=window_size,
            imsize=(aa,bb),
            threshold=theta,
            group_size=group_size
        )
        D, indices = zip(*pool.map(lowrank_i, np.arange(patches.shape[1])))
        D = np.array(D)
        indices = np.array(indices, dtype=np.int)
        pool.close()
        pool.join()

        D += np.einsum('ik,j->ijk', patch_means[indices], np.ones(patches.shape[0]))

        if s > 0:
            # ----- Sparse Coding -----
            patches_3d = np.zeros((patches.shape[0]*group_size_s, patches.shape[1]))

            for i in range(group_size_s):
                patches_3d[i*patches.shape[0]:(i+1)*patches.shape[0], :] = patches[
                    :, indices[:, i]
                ]

            sparse_codes = transform @ patches_3d
            #FIXME - implement variable threshold for each image
            # sparse_indices = (sparse_codes > recon.lam).astype(np.int)
            # if not sparse_indices.any():
            #     sparse_codes = np.zeros_like(sparse_codes)
            #
            # sparse_codes = sparse_codes * sparse_indices

            for i in range(num_sources):
                ind = np.arange(i*aa*bb, (i+1)*aa*bb)
                sparse_codes[:,ind] = hard_thresholding(
                    sparse_codes[:,ind],
                    threshold = np.sqrt(2*lam[i] / s)
                )

            if learning is True:
                u, s, v_T = np.linalg.svd(sparse_codes @ patches_3d.T)
                transform = u @ v_T

            # ----- Image Update -----
            Whb1 = np.zeros_like(patches)
            Whb = transform.T @ sparse_codes
            for i in range(group_size_s):
                Whb1 = indsum(Whb1, Whb[i*patches.shape[0]:(i+1)*patches.shape[0], :], indices[:, i])

            indvals = np.array(list(Counter(indices[:, :group_size_s].flatten()).values()))
            indkeys = np.argsort(np.array(list(Counter(indices[:, :group_size_s].flatten()).keys())))
            Whb1 = Whb1 / indvals[indkeys]

            Fc = np.fft.fft2(
                patch_aggregator(
                    Whb1,
                    patch_shape=patch_shape,
                    image_shape=(num_sources,1,aa,bb)
                )
            )
        else:
            Fc = np.zeros_like(psfdfts_h_meas)

        if lr > 0:
            VhD = np.zeros_like(patches)
            for i in range(group_size):
                VhD = indsum(VhD, D[:,:,i].T, indices[:, i])

            indvals = np.array(list(Counter(indices.flatten()).values()))
            indkeys = np.argsort(np.array(list(Counter(indices.flatten()).keys())))
            VhD = VhD / indvals[indkeys]

            Fd = np.fft.fft2(
                patch_aggregator(
                    VhD,
                    patch_shape=patch_shape,
                    image_shape=(num_sources,1,aa,bb)
                )
            )
        else:
            Fd = np.zeros_like(psfdfts_h_meas)


        recon = np.real(
            np.fft.ifft2(
                block_mul(
                    SIG_inv,
                    s * Fc + lr * Fd + psfdfts_h_meas
                )
            )
        )

        if iter % 1 == 0 or iter == iternum - 1:
            ssim1 = np.zeros(num_sources)
            mse1 = np.mean((sources - recon)**2, axis=(1, 2, 3))
            psnr1 = 20 * np.log10(np.max(sources, axis=(1,2,3))/np.sqrt(mse1))
            for i in range(num_sources):
                ssim1[i] = ssim(sources[i,0], recon[i,0],
                    data_range=np.max(recon[i,0])-np.min(recon[i,0]))
            plotter4d(recon,
                cmap='gist_heat',
                fignum=3,
                figsize=(5.6,8),
                title='Iteration: {}\n Recon. SSIM={}\n Recon. PSNR={}'.format(iter, ssim1, psnr1)
            )
            plt.pause(0.5)

    return recon

def admm(
        *,
        sources,
        psfs,
        measurements,
        regularizer,
        recon_init_method,
        iternum,
        nu,
        **kwargs
):
    """Function that implements PSSI deconvolution with the ADMM algorithm using
    the specified regularization method.

    Args:
        sources (ndarray): 4d array of sources
        psfs (PSFs): PSFs object containing psfs and related data
        measurements (ndarray): 4d array of measurements
        regularizer (str): string that specifies the type of regularization
        recon_init_method (str): ['zeros' or 'tikhonov'] string specifying the
            initialization method for the reconstructions
        iternum (int): number of iterations of ADMM
        nu (float): augmented Lagrangian parameter (step size) of ADMM
        kwargs (dict): keyword arguments to be passed to the regularizers

    Returns:
        ndarray of reconstructed images
    """

    [k,num_sources,aa,bb] = psfs.selected_psfs.shape[:2] + sources.shape[2:]
    if 'lam' in kwargs.keys():
        if type(kwargs['lam']) is np.float:
            kwargs['lam'] = np.ones(num_sources) * kwargs['lam']

    ################## initialize the primal/dual variables ##################
    if recon_init_method is 'zeros':
        primal1 = np.zeros((num_sources,1,aa,bb))

    elif recon_init_method is 'tikhonov':
        primal1 = tikhonov(
            sources=sources,
            psfs=psfs,
            measurements=measurements,
            tikhonov_lam=kwargs['tikhonov_lam'],
            tikhonov_order=kwargs['tikhonov_order']
        )

    primal2, kwargs = primal2_init(regularizer=regularizer, primal1=primal1,
        psfs=psfs, **kwargs)
    # primal2 = np.zeros_like(primal2) # uncomment if zero initialization is not desired
    dual = primal2.copy()

    ################# pre-compute some arrays for efficiency #################
    psfdfts_h_meas = block_mul(
        psfs.selected_psf_dfts_h,
        np.fft.fft2(measurements)
    ) # this is reshaped FA^Ty term where F is DFT matrix
    SIG_inv = get_SIG_inv(regularizer=regularizer, psfs=psfs, nu=nu, **kwargs)

    for iter in range(iternum):
        ######################### PRIMAL1 UPDATE #########################
        primal1, kwargs = primal1_update(regularizer=regularizer, psfs=psfs,
            psfdfts_h_meas=psfdfts_h_meas, SIG_inv=SIG_inv, primal1=primal1,
            primal2=primal2, dual=dual, nu=nu, **kwargs)

        ######################### PRIMAL2 UPDATE #########################
        primal2, pre_primal2, kwargs = primal2_update(regularizer=regularizer,
            primal1=primal1, psfs=psfs, primal2=primal2, dual=dual,
            nu=nu, **kwargs)

        ########################### DUAL UPDATE ###########################
        dual += (pre_primal2 - primal2)

        if (regularizer is 'sparsepatch') and (kwargs['learning'] is True):
            u,s,vT = np.linalg.svd((primal2-dual) @ kwargs['patches'].T)
            kwargs['transform'] = u @ vT

        if iter % 5 == 0 or iter == iternum - 1:
            # print(dfid, l1_reg2, dfid+l1_reg2)
            ssim1 = np.zeros(num_sources)
            mse1 = np.mean((sources - primal1)**2, axis=(1, 2, 3))
            psnr1 = 20 * np.log10(np.max(sources, axis=(1,2,3))/np.sqrt(mse1))
            for i in range(num_sources):
                ssim1[i] = ssim(sources[i,0], primal1[i,0],
                    data_range=np.max(primal1[i,0])-np.min(primal1[i,0]))
            plotter4d(primal1,
                cmap='gist_heat',
                fignum=3,
                figsize=(5.6,8),
                title='Iteration: {}\n Recon. SSIM={}\n Recon. PSNR={}'.format(iter, ssim1, psnr1)
            )
            plt.pause(0.5)

    return primal1

def tikhonov(
            *,
            sources,
            psfs,
            measurements,
            tikhonov_lam,
            tikhonov_order
):
    """Perform Tikhonov regularization based image reconstruction for PSSI.

    Solves x_hat = argmin_{x} { ||Ax-y||_2^2 + lam * ||Dx||_2^2 }. D is the
    discrete derivative operator of order `tikhonov_order`.

    Args:
        sources (ndarray): 4d array of ground truth spectral images
        psfs (PSFs): PSFs object containing psfs and other csbs state data
        measured_noisy (ndarray): 4d array of noisy measurements
        tikhonov_lam (float): regularization parameter of tikhonov
        tikhonov_order (int): [0,1 or 2] order of the discrete derivative
            operator used in the tikhonov regularization

    Returns:
        4d array of the reconstructed images
    """
    [k,num_sources,aa,bb] = psfs.selected_psfs.shape[:2] + sources.shape[2:]
    # DFT of the kernel corresponding to (D^TD)
    LAM = get_LAM(rows=aa,cols=bb,order=tikhonov_order)
    return np.real(
        np.fft.ifft2(
                block_mul(
                    block_inv(
                        psfs.selected_GAM +
                        tikhonov_lam * np.einsum('ij,kl', np.eye(num_sources), LAM)
                    ),
                    block_mul(
                        psfs.selected_psf_dfts_h,
                        np.fft.fft2(measurements)
                    )
                )
        )
    )

def get_SIG_inv(
                *,
                regularizer,
                psfs,
                nu,
                **kwargs
):
    """ The function that returns the spectrum of (A^TA+nu*W^TW + ...)^{-1}

    Args:
        regularizer (string): string that specifies the regularization
        psfs (PSFs): PSFs object containing psfs and other csbs state data
        kwargs (dict): keyword arguments of parameters

    Returns:
        ndarray of the spectrum of (A^TA+nu*W^TW + ...)^{-1}
    """
    [_,num_sources,aa,bb] = psfs.selected_psfs.shape
    psize = np.size(np.empty(kwargs['patch_shape']))
    if regularizer is 'TV':
        LAM = get_LAM(rows=aa,cols=bb,order=1)
        spectrum = nu * np.einsum('ij,kl->ijkl', np.eye(num_sources), LAM)

    elif regularizer is 'patch_based':
        LAM = psize * np.ones((aa,bb))
        spectrum = nu * np.einsum('ij,kl->ijkl', np.eye(num_sources), LAM)

    elif regularizer is 'lowrank':
        LAM = psize * np.ones((aa,bb))
        spectrum = nu * np.einsum('ij,kl->ijkl', np.eye(num_sources), LAM)

    elif regularizer is 'bm3d_pnp':
        LAM = np.ones((aa,bb))
        spectrum = nu * np.einsum('ij,kl->ijkl', np.eye(num_sources), LAM)

    elif regularizer is 'dncnn':
        LAM = np.ones((aa,bb))
        spectrum = nu * np.einsum('ij,kl->ijkl', np.eye(num_sources), LAM)

    return block_inv(psfs.selected_GAM + spectrum)


def primal1_update(
                *,
                regularizer,
                psfs,
                primal1,
                primal2,
                dual,
                psfdfts_h_meas,
                SIG_inv,
                nu,
                **kwargs
):
    """Function that updates the first primal variable of ADMM.

    Args:
        regularizer (string): string that specifies the regularization type
        psfs (PSFs): PSFs object containing psfs and other csbs state data
        primal1 (ndarray): first primal variable of ADMM
        primal2 (ndarray): second primal variable of ADMM
        dual (ndarray): dual variable of ADMM
        psfdfts_h_meas (ndarray): FA^Ty term where F is DFT matrix
        SIG_inv (ndarray): spectrum of (A^TA+nu*W^TW + ...)^{-1}
        nu (float): augmented Lagrangian parameter of ADMM (step size)
        kwargs (dict): keyword arguments of parameters related to the regularizer

    Returns:
        ndarray of updated primal1
        kwargs with possible modification
    """
    [_,num_sources,aa,bb] = psfs.selected_psfs.shape
    if regularizer is 'TV':
        pre_primal1 = diff_T(primal2 - dual)

    elif regularizer is 'patch_based':
        pre_primal1 = patch_aggregator(
            kwargs['transform'].conj().T @ (primal2 - dual),
            patch_shape = kwargs['patch_shape'],
            image_shape = (num_sources, 1, aa, bb)
        )

    elif regularizer is 'lowrank':
        psize = np.size(np.empty(kwargs['patch_shape']))
        pre_primal1 = np.zeros((psize, aa*bb*num_sources))
        for i in range(kwargs['group_size']):
            pre_primal1 = indsum(
                pre_primal1,
                (primal2[:,:,i] - dual[:,:,i]).T,
                kwargs['indices'][:, i]
            )

        indvals = np.array(list(Counter(kwargs['indices'].flatten()).values()))
        indkeys = np.argsort(np.array(list(Counter(kwargs['indices'].flatten()).keys())))
        pre_primal1 = pre_primal1 / indvals[indkeys]

        pre_primal1 = patch_aggregator(
            pre_primal1,
            patch_shape = kwargs['patch_shape'],
            image_shape=(num_sources, 1, aa, bb)
        )

    elif regularizer is 'bm3d_pnp':
        pre_primal1 = primal2 - dual

    elif regularizer is 'dncnn':
        pre_primal1 = primal2 - dual

    return np.real(
        np.fft.ifft2(
            block_mul(
                SIG_inv,
                psfdfts_h_meas + nu * np.fft.fft2(pre_primal1)
            )
        )
    ), kwargs


def primal2_update(
                *,
                regularizer,
                psfs,
                primal1,
                primal2,
                dual,
                nu,
                **kwargs
):
    """Function that updates the second primal variable of ADMM.

    Args:
        regularizer (string): string that specifies the regularization type
        psfs (PSFs): PSFs object containing psfs and other csbs state data
        primal1 (ndarray): first primal variable of ADMM
        primal2 (ndarray): second primal variable of ADMM
        dual (ndarray): dual variable of ADMM
        nu (float): augmented Lagrangian parameter of ADMM (step size)
        kwargs (dict): keyword arguments of parameters related to the regularizer

    Returns:
        primal2 (ndarray): ndarray of updated primal2
        pre_primal2 (ndarray): ndarray of domain transformed primal1 variable (Wx)
        kwargs with possible modification
    """
    [_,num_sources,aa,bb] = psfs.selected_psfs.shape
    if regularizer is 'TV':
        pre_primal2 = diff(primal1)
        for i in range(num_sources):
            ind = np.arange(i*aa*bb, (i+1)*aa*bb)
            primal2[:,ind] = hard_thresholding(
                pre_primal2[:,ind] + dual[:,ind],
                threshold = np.sqrt(2*kwargs['lam'][i] / nu)
            )

    elif regularizer is 'patch_based':
        kwargs['patches'] = patch_extractor(
            primal1, patch_shape = kwargs['patch_shape']
        )
        pre_primal2 = kwargs['transform'] @ kwargs['patches']
        for i in range(num_sources):
            ind = np.arange(i*aa*bb, (i+1)*aa*bb)
            primal2[:,ind] = hard_thresholding(
                pre_primal2[:,ind] + dual[:,ind],
                threshold = np.sqrt(2*kwargs['lam'][i] / nu)
            )

    elif regularizer is 'lowrank':
        patches = patch_extractor(
            primal1, patch_shape = kwargs['patch_shape']
        )
        patch_means = np.mean(patches, axis=0)
        dual_means = np.mean(dual, axis=1)
        patches_zeromean = (patches - patch_means).T

        pool = multiprocessing.Pool()
        block_matching_i = functools.partial(block_matching,
            patches_zeromean=patches_zeromean,
            window_size=kwargs['window_size'],
            imsize=(aa,bb),
            group_size=kwargs['group_size']
        )
        pre_primal2, indices = zip(*pool.map(block_matching_i, np.arange(patches.shape[1])))
        pool.close()
        pool.join()
        pre_primal2 = np.array(pre_primal2)
        indices = np.array(indices, dtype=np.int)
        dual0 = dual - np.einsum('ik,j->ijk', dual_means, np.ones(dual.shape[1]))
        blocks = pre_primal2 + dual0

        # pool = multiprocessing.Pool()
        # lowranker_i = functools.partial(lowranker,
        #     blocks=blocks,
        #     imsize=(aa,bb),
        #     threshold=kwargs['lam']
        # )
        # primal2 = zip(*pool.map(lowranker_i, np.arange(patches.shape[1])))
        # pool.close()
        # pool.join()
        # primal2 = np.array(primal2)1
        for i in range(patches.shape[1]):
            # find out which image the index i correspond to (to apply threshold accordingly)
            im = np.int(i / (aa*bb))

            u, s, v_T = np.linalg.svd(blocks[i], full_matrices=False)
            primal2[i] = u @ np.diag(hard_thresholding(s, threshold=kwargs['lam'][im])) @ v_T
        pre_primal2 += np.einsum(
            'ik,j->ijk', patch_means[indices], np.ones(patches.shape[0])
        )
        primal2 += np.einsum(
            'ik,j->ijk', patch_means[indices], np.ones(patches.shape[0])
        ) + np.einsum('ik,j->ijk', dual_means, np.ones(dual.shape[1]))
        kwargs['indices'] = indices


    elif regularizer is 'bm3d_pnp':
        pre_primal2 = primal1
        for i in range(num_sources):
            primal2[i,0] = pybm3d.bm3d.bm3d(primal1[i,0]+dual[i,0], np.sqrt(kwargs['lam'][i]/nu))


    elif regularizer is 'dncnn':
        pre_primal2 = primal1
        for i in range(num_sources):
            noisy = np.reshape(primal1[i,0]+dual[i,0], (1,aa,bb,1))
            primal2[i,0] = kwargs['model'].predict(noisy)[0,:,:,0]

    return primal2, pre_primal2, kwargs

def primal2_init(
                *,
                regularizer,
                primal1,
                psfs,
                **kwargs
):
    """Function that initializes the second primal variable of ADMM.

    Args:
        regularizer (string): string that specifies the regularization type
        primal1 (ndarray): first primal variable of ADMM
        psfs (PSFs): PSFs object containing psfs and other csbs state data
        kwargs (dict): keyword arguments of parameters related to the regularizer

    Returns:
        primal2 (ndarray): ndarray of initialized primal2
        kwargs with possible modification
    """
    [_,num_sources,aa,bb] = psfs.selected_psfs.shape
    if regularizer is 'TV':
        primal2 = diff(primal1)

    elif regularizer is 'patch_based':
        patches = patch_extractor(
            primal1, patch_shape = kwargs['patch_shape']
        )
        primal2 = kwargs['transform'] @ patches

    elif regularizer is 'lowrank':
        patches = patch_extractor(
            primal1, patch_shape = kwargs['patch_shape']
        )
        patch_means = np.mean(patches, axis=0)
        patches_zeromean = (patches - patch_means).T
        indices = np.zeros((patches.shape[1], kwargs['group_size']), dtype=np.int)

        pool = multiprocessing.Pool()
        block_matching_i = functools.partial(block_matching,
            patches_zeromean=patches_zeromean,
            window_size=kwargs['window_size'],
            imsize=(aa,bb),
            group_size=kwargs['group_size']
        )
        primal2, indices = zip(*pool.map(block_matching_i, np.arange(patches.shape[1])))
        primal2 = np.array(primal2)
        indices = np.array(indices, dtype=np.int)
        pool.close()
        pool.join()
        primal2 += np.einsum(
            'ik,j->ijk', patch_means[indices], np.ones(patches.shape[0])
        )
        kwargs['indices'] = indices

    elif regularizer is 'bm3d_pnp':
        primal2 = np.zeros_like(primal1)

    elif regularizer is 'dncnn':
        primal2 = np.zeros_like(primal1)

    return primal2, kwargs
