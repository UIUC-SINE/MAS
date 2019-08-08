import numpy as np
from mas.block import block_mul, block_inv
from mas.deconvolution.common import get_LAM
from mas.forward_model import size_equalizer
from mas.block import block_mul, block_herm

def tikhonov(
            *,
            psfs,
            measurements,
            tikhonov_lam=0.129,
            tikhonov_order=1
):
    """Perform Tikhonov regularization based image reconstruction for PSSI.

    Solves x_hat = argmin_{x} { ||Ax-y||_2^2 + lam * ||Dx||_2^2 }. D is the
    discrete derivative operator of order `tikhonov_order`.

    Args:
        psfs (PSFs): PSFs object containing psfs and other csbs state data
        measured_noisy (ndarray): 4d array of noisy measurements
        tikhonov_lam (float): regularization parameter of tikhonov
        tikhonov_order (int): [0,1 or 2] order of the discrete derivative
            operator used in the tikhonov regularization

    Returns:
        4d array of the reconstructed images
    """
    num_sources = psfs.psfs.shape[1]
    rows, cols = measurements.shape[1:]
    psfs.psf_dfts = np.repeat(
            np.fft.fft2(size_equalizer(psfs.psfs, ref_size=[rows,cols])),
            psfs.copies.astype(int), axis=0
    )
    psfs.psf_GAM = block_mul(
        block_herm(psfs.psf_dfts),
        psfs.psf_dfts
    )
    # DFT of the kernel corresponding to (D^TD)
    LAM = get_LAM(rows=rows,cols=cols,order=tikhonov_order)
    return np.real(
        np.fft.ifft2(
                block_mul(
                    block_inv(
                        psfs.psf_GAM +
                        tikhonov_lam * np.einsum('ij,kl', np.eye(num_sources), LAM)
                    ),
                    block_mul(
                        block_herm(psfs.psf_dfts),
                        np.fft.fft2(np.fft.fftshift(measurements, axes=(1,2)))
                    )
                )
        )
    )
