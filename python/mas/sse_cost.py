#!/usr/bin/env python3
# Ulas Kamaci - 2018-04-02

import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fft2, fftshift, ifftshift
from mas.block import block_mul, block_inv, block_herm
from mas.deconvolution.tikhonov import get_LAM

def diff_matrix(size):
    """Create discrete derivative approximation matrix

    Returns a discrete derivative approximation matrix in the x direction

    e.g. for size=5

    |  1 -1  0  0  0 |
    |  0  1 -1  0  0 |
    |  0  0  1 -1  0 |
    |  0  0  0  1 -1 |
    | -1  0  0  0  1 |

    Args:
        size (int): length of a side of this matrix

    Returns:
        (ndarray): array of dimension (size, size)
    """

    return np.eye(size) - np.roll(np.eye(size), -1, axis=0)


def init(psfs, **kwargs):
    """
    """
    _, _, rows, cols = psfs.psfs.shape
    # psf_dfts = np.fft.fft2(psfs.psfs, axes=(2, 3))
    psf_dfts = psfs.psf_dfts

    initialized_data = {
        "psf_dfts": psf_dfts,
        "GAM": block_mul(
            block_herm(
                # scale rows of psf_dfts by copies
                # split across sqrt to prevent rounding error
                np.einsum(
                    'i,ijkl->ijkl', np.sqrt(psfs.copies),
                    psf_dfts
                )
            ),
            np.einsum(
                'i,ijkl->ijkl', np.sqrt(psfs.copies),
                psf_dfts
            ),
        ),
        "LAM": get_LAM(rows=rows,cols=cols,order=kwargs['order'])
    }

    psfs.initialized_data = initialized_data


def iteration_end(psfs, lowest_psf_group_index):
    """
    """
    psfs.initialized_data['GAM'] -= block_mul(
        block_herm(psfs.initialized_data['psf_dfts'][lowest_psf_group_index:lowest_psf_group_index + 1]),
        psfs.initialized_data['psf_dfts'][lowest_psf_group_index:lowest_psf_group_index + 1]
    )


def SIG_e_dft(psfs, lam):
    """Compute SIG_e_dft for given PSFs

    Args:
        psfs (PSFs): psf object
        lam (float): regularization parameter
    """

    _, num_sources, _, _ = psfs.psfs.shape

    return (
        psfs.initialized_data['GAM'] +
        lam * np.einsum('ij,kl', np.eye(num_sources), psfs.initialized_data['LAM'])
    )


def cost(psfs, psf_group_index, **kwargs):
    """
    """

    iteration_SIG_e_dft = (
        SIG_e_dft(psfs, kwargs['lam']) -
        block_mul(
            block_herm(psfs.initialized_data['psf_dfts'][psf_group_index:psf_group_index + 1]),
            psfs.initialized_data['psf_dfts'][psf_group_index:psf_group_index + 1]
        )
    )

    return np.real(np.sum(np.trace(block_inv(iteration_SIG_e_dft))))
