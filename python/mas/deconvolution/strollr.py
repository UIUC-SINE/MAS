import numpy as np
from mas.deconvolution.common import indsum
from mas.deconvolution.common import patch_extractor, patch_aggregator
from skimage.measure import compare_ssim
from mas.plotting import plotter4d
from mas.block import block_mul, block_inv
import functools
import multiprocessing

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
                ssim1[i] = compare_ssim(sources[i,0], recon[i,0],
                    data_range=np.max(recon[i,0])-np.min(recon[i,0]))
            plotter4d(recon,
                cmap='gist_heat',
                fignum=3,
                figsize=(5.6,8),
                title='Iteration: {}\n Recon. SSIM={}\n Recon. PSNR={}'.format(iter, ssim1, psnr1)
            )
            plt.pause(0.5)

    return recon
