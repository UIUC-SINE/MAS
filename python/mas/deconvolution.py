#!/usr/bin/env python3
# Ulas Kamaci 2018-08-27

import numpy as np
import logging
from matplotlib import pyplot as plt
from mas.sse_cost import block_mul, block_inv, block_herm, SIG_e_dft, get_LAM
from mas.forward_model import size_equalizer

logging.basicConfig(level=logging.INFO, format='%(message)s')

def tikhonov(recon, *, psfs, measurements, SIG_e_dft_inv, **kwargs):
    """Perform Tikhonov regularization based image reconstruction for PSSI.

    Args:
        psfs (PSFs): PSFs object containing psfs and other csbs state data
        measured_noisy (ndarray): 4d array of noisy measurements
        SIG_e_dft_inv (ndarray): 4d array of inverse of the SIG_e_dft matrix

    Returns:
        4d array of reconstructed images
    """

    recon.reconstructed.append(
        np.real(
            np.fft.ifft2(
                    block_mul(
                        SIG_e_dft_inv,
                        block_mul(
                            psfs.selected_psf_dfts_h,
                            np.fft.fft2(measurements)
                        )
                    )
            )
        )
    )


def patch_extractor(image, patch_shape):
    """Create a patch matrix where each column is a vectorized patch.
    It works with both 2D and 3D patches. Patches at the boundaries are
    extrapolated as if the image is periodically replicated. This way all
    the patches have the same dimension.

    Args:
        image (ndarray): 3D matrix representing the data cube
        patch_shape (ndarray): 1D array of length 3

    Returns:
        patch_mtx (ndarray): patch matrix containing vectorized patches in its
        columns. Number of columns (patches) is equal to the number of pixels
        (or voxels) in the image.
    """

    assert len(patch_shape) == 3, 'patch_shape must have length=3'

    [aa,bb,p] = image.shape
    patch_size = 1
    for i in patch_shape:
        patch_size = patch_size * i
    patch_mtx = np.zeros((patch_size, np.size(image)))

    # periodically extend the input image
    temp = np.concatenate((image, image[:patch_shape[0] - 1,:,:]), axis = 0)
    temp = np.concatenate((temp, temp[:,:patch_shape[1] - 1,:]), axis = 1)
    temp = np.concatenate((temp, temp[:,:,:patch_shape[2] - 1]), axis = 2)
    [rows, cols, slices] = np.unravel_index(
    range(patch_size), patch_shape)
    for i in range(np.size(patch_shape)):
        patch_mtx[i,:] = np.reshape(temp[rows[i] : aa + rows[i],
        cols[i] : bb + cols[i], slices[i] : p + slices[i]], -1)

    return patch_mtx


def patch_aggregator(patch_mtx, patch_shape = None, image_shape = None):
    """Implements the adjoint of the patch extractor operator.

    Args:
        patch_mtx (ndarray): patch matrix containing vectorized patches in its
        columns. Number of columns (patches) is equal to the number of pixels
        (or voxels) in the image.
        patch_shape (ndarray): 1D array of length 3
        image_shape (ndarray): 1D array of length 3

    Returns:
        image (ndarray): 3D matrix consisting of the aggregated patches
    """
    temp = np.zeros(image_shape + patch_shape - 1)

    [rows, cols, slices] = np.unravel_index(
    range(patch_mtx.shape[0]), patch_shape)

    for i in range(patch_mtx.shape[0]):
        temp[rows[i] : image_shape[0] + rows[i],
        cols[i] : image_shape[1] + cols[i],
        slices[i] : image_shape[2]  + slices[i]] = temp[
        rows[i] : image_shape[0] + rows[i],
        cols[i] : image_shape[1] + cols[i],
        slices[i] : image_shape[2]  + slices[i]] + np.reshape(
        patch_mtx[i,:], image_shape)

    temp[:,:,:patch_shape[2] - 1] = temp[:,:,:patch_shape[2] - 1] + temp[
    :,:,image_shape[2]:]
    temp[:,:patch_shape[1] - 1,:] = temp[:,:patch_shape[1] - 1,:] + temp[
    :,image_shape[1]:,:]
    temp[:patch_shape[0] - 1,:,:] = temp[:patch_shape[0] - 1,:,:] + temp[
    image_shape[0]:,:,:]

    return temp[:image_shape[0], :image_shape[1], :image_shape[2]]

# def tldecon_p3d(*, image, psfs, meas, patch_size, transform, nu, tr_reg, maxiter, ):

class Reconstruction():
    """A class for holding the reconstructions and related parameters

    Args:
        sources (ndarray): 4d array of sources
        measurements (ndarray): 5d array of measurements
        psfs (PSFs): PSFs object containing psfs and related data
        deconvolver (def): function to perform deconvolution (default, mas.pssi_deconvolution.tikhonov)
    """

    def __init__(
        self,
        *,
        psfs,
        sources,
        measurements,
        deconvolver=tikhonov,
        **kwargs
    ):

        num_instances = measurements.shape[0]

        # ----- precompute fixed parameters for fast implementation -----
        pre_computed = {}

        if deconvolver is tikhonov:
            [_,num_sources,aa,bb] = psfs.selected_psfs.shape
            LAM = get_LAM(rows=aa,cols=bb,order=kwargs['order'])

            pre_computed['SIG_e_dft_inv'] = block_inv(
                block_mul(
                    psfs.selected_psf_dfts_h,
                    psfs.selected_psf_dfts
                ) +
                kwargs['lam'] * np.einsum('ij,kl', np.eye(num_sources), LAM)
            )

        # ----- run the deconvolution algorithm for all the noise realizations -----
        self.reconstructed = []
        for i in range(num_instances):
            logging.info('{}/{}'.format(i+1, num_instances))
            deconvolver(
                self,
                measurements=measurements[i],
                psfs=psfs,
                **pre_computed,
                **kwargs
            )
        self.reconstructed = np.array(self.reconstructed)

        self.mse = []
        for i, recon in enumerate(self.reconstructed):
            self.mse.append(np.mean((sources - recon)**2, axis=(1, 2, 3)))
        self.mse = np.array(self.mse)
        self.mse_average = np.mean(self.mse, axis=0)
