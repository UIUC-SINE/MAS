import numpy as np
from mas.deconvolution.common import patch_extractor, patch_aggregator
from skimage.measure import compare_ssim
from mas.plotting import plotter4d
from mas.block import block_mul, block_inv
import functools
import multiprocessing
import pybm3d

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
                ssim1[i] = compare_ssim(sources[i,0], primal1[i,0],
                    data_range=np.max(primal1[i,0])-np.min(primal1[i,0]))
            plotter4d(primal1,
                cmap='gist_heat',
                fignum=3,
                figsize=(5.6,8),
                title='Iteration: {}\n Recon. SSIM={}\n Recon. PSNR={}'.format(iter, ssim1, psnr1)
            )
            plt.pause(0.5)

    return primal1

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
