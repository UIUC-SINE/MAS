import numpy as np
from mas.deconvolution.common import patch_extractor, patch_aggregator
from skimage.measure import compare_ssim
from mas.plotting import plotter4d
from mas.block import block_mul, block_inv

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
