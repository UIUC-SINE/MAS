import numpy as np
from mas.deconvolution.common import deconv_plotter, hard_thresholding
from mas.deconvolution.common import patch_extractor, patch_aggregator
from skimage.metrics import structural_similarity as compare_ssim
from mas.block import block_mul, block_inv

def sparsepatch(
            *,
            sources,
            psfs,
            measurements,
            recon_init,
            iternum,
            plot=True,
            periter,
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
        recon_init (ndarray): initialization for the reconstructed image(s)
        iternum (int): number of iterations of ADMM
        plot (boolean): if set to True, display the reconstructions as the iterations go
        periter (int): iteration period of displaying the reconstructions
        nu (float): augmented Lagrangian parameter
        lam (float): penalty parameter of the sparsity term
        patch_shape (tuple): tuple of the shape of the patches used
        transform (ndarray): ndarray of the sparsifying transform used
        learning (bool): boolean variable of whether the transform gets updated
    """
    num_sources = psfs.psfs.shape[1]
    rows, cols = measurements.shape[1:]
    psize = np.size(np.empty(patch_shape))
    # mse_inner = np.zeros((num_sources,recon.maxiter))
    if type(lam) is np.float or type(lam) is np.int:
        lam = np.ones(num_sources) * lam

    ################## initialize the reconstruction ##################
    recon = recon_init

    ################# pre-compute some arrays for efficiency #################
    psfs.psf_dfts = np.repeat(
            np.fft.fft2(size_equalizer(psfs.psfs, ref_size=[rows,cols])),
            psfs.copies.astype(int), axis=0
    )
    psfs.psf_GAM = block_mul(
        block_herm(psfs.psf_dfts),
        psfs.psf_dfts
    )
    psfdfts_h_meas = block_mul(
        block_herm(psfs.psf_dfts),
        np.fft.fft2(np.fft.fftshift(measurements, axes=(1,2)))
    ) # this is reshaped FA^Ty term where F is DFT matrix
    LAM = psize * np.ones((rows,cols))
    spectrum = nu * np.einsum('ij,kl->ijkl', np.eye(num_sources), LAM)
    SIG_inv = block_inv(psfs.psf_GAM + spectrum)

    for iter in range(iternum):

        # ----- Sparse Coding -----
        patches = patch_extractor(
            recon,
            patch_shape=patch_shape
        )

        sparse_codes = transform @ patches
        for i in range(num_sources):
            sparse_codes[:,i*rows*cols:(i+1)*rows*cols] = hard_thresholding(
                sparse_codes[:,i*rows*cols:(i+1)*rows*cols],
                threshold=np.sqrt(lam[i]/nu)
            )

        # ----- Image Update -----
        Fc = np.fft.fft2(
            patch_aggregator(
                transform.conj().T @ sparse_codes,
                patch_shape=patch_shape,
                image_shape=(num_sources,rows,cols)
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


        if plot==True and ((iter+1) % periter == 0 or iter == iternum - 1):
            deconv_plotter(sources=sources, recons=recon, iter=iter)

    return recon
