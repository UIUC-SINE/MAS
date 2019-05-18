import numpy as np
import matplotlib.pyplot as plt
from mas.deconvolution.common import patch_extractor, patch_aggregator, dctmtx, hard_thresholding
from mas.deconvolution import tikhonov
from skimage.measure import compare_ssim
from mas.deconvolution.common import deconv_plotter, get_LAM
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
        recon_init,
        iternum,
        periter,
        nu,
        lam,
        **kwargs
):
    """Function that implements PSSI deconvolution with the ADMM algorithm using
    the specified regularization method.

    Args:
        sources (ndarray): 3d array of sources
        psfs (PSFs): PSFs object containing psfs and related data
        measurements (ndarray): 3d array of measurements
        regularizer (function): function that specifies the regularization type
        recon_init (ndarray): initialization for the reconstructed image(s)
        iternum (int): number of iterations of ADMM
        periter (int): iteration period of displaying the reconstructions
        nu (float): augmented Lagrangian parameter (step size) of ADMM
        lam (list): regularization parameter of dimension num_sources
        kwargs (dict): keyword arguments to be passed to the regularizers

    Returns:
        ndarray of reconstructed images
    """
    k, num_sources = psfs.selected_psfs.shape[:2]
    aa, bb = sources.shape[1:]

    ################## initialize the primal/dual variables ##################
    primal1 = recon_init
    primal2 = None
    dual = None

    ################# pre-compute some arrays for efficiency #################
    psfdfts_h_meas = block_mul(
        psfs.selected_psf_dfts_h,
        np.fft.fft2(np.fft.fftshift(measurements, axes=(1,2)))
    ) # this is reshaped FA^Ty term where F is DFT matrix
    SIG_inv = get_SIG_inv(regularizer=regularizer, psfs=psfs, nu=nu, **kwargs)

    for iter in range(iternum):
        ######################### PRIMAL 1,2 UPDATES #########################
        primal1, primal2, pre_primal2, dual = regularizer(psfs=psfs,
            psfdfts_h_meas=psfdfts_h_meas, SIG_inv=SIG_inv, primal1=primal1,
            primal2=primal2, dual=dual, nu=nu, lam=lam, **kwargs)

        ########################### DUAL UPDATE ###########################
        dual += (pre_primal2 - primal2)

        if (iter+1) % periter == 0 or iter == iternum - 1:
            deconv_plotter(sources=sources, recons=primal1, iter=iter)

    return primal1

def patch_based(
    *,
    psfs,
    primal1,
    primal2,
    dual,
    psfdfts_h_meas,
    SIG_inv,
    nu,
    lam,
    patch_shape=(6,6,1),
    transform=dctmtx((6,6,1)),
    learning=True,
    **kwargs
):
    """Function that updates the first and the second primal variables of ADMM
    based on the patch based sparsifying transform learning regularization.

    Args:
        psfs (PSFs): PSFs object containing psfs and other csbs state data
        primal1 (ndarray): first primal variable of ADMM
        primal2 (ndarray): second primal variable of ADMM
        dual (ndarray): dual variable of ADMM
        psfdfts_h_meas (ndarray): FA^Ty term where F is DFT matrix
        SIG_inv (ndarray): spectrum of (A^TA+nu*W^TW + ...)^{-1}
        nu (float): augmented Lagrangian parameter of ADMM (step size)
        lam (list): regularization parameter of dimension num_sources
        patch_shape (tuple): shape of the used patches in the method
        transform (ndarray): initialization for the sparsifying transform matrix
        learning (bool): the boolian parameter specifying if the transform is fixed
            to the initialized value, or gets updated over the iterations
        kwargs (dict): keyword arguments of parameters related to the regularizer
    Returns:
        ndarray of updated primal1
        ndarray of updated primal2
        ndarray of updated intermediate variable pre_primal2
        ndarray of initialized dual
    """
    [_,num_sources,aa,bb] = psfs.selected_psfs.shape
    ###################  INITIALIZE PRIMAL2 and DUAL ##################
    if primal2 is None:
        patches = patch_extractor(primal1, patch_shape=patch_shape)
        primal2 = transform @ patches
        dual = primal2.copy()

    ######################### PRIMAL1 UPDATE #########################
    pre_primal1 = patch_aggregator(
       transform.conj().T @ (primal2 - dual),
        patch_shape = patch_shape,
        image_shape = (num_sources, aa, bb)
    )
    primal1 = primal1_update_gaussian(SIG_inv=SIG_inv, ATy=psfdfts_h_meas,
        nu=nu, pre_primal1=pre_primal1)

    ######################### PRIMAL2 UPDATE #########################
    patches = patch_extractor(
        primal1, patch_shape = patch_shape
    )
    pre_primal2 = transform @ patches
    for i in range(num_sources):
        ind = np.arange(i*aa*bb, (i+1)*aa*bb)
        primal2[:,ind] = hard_thresholding(
            pre_primal2[:,ind] + dual[:,ind],
            threshold = np.sqrt(lam[i] / nu)
        )

    ######################### TRANSFORM UPDATE #########################
    if learning is True:
        u,s,vT = np.linalg.svd((primal2-dual) @ patches.T)
        transform = u @ vT

    return primal1, primal2, pre_primal2, dual


def TV(
    *,
    psfs,
    primal1,
    primal2,
    dual,
    psfdfts_h_meas,
    SIG_inv,
    nu,
    lam,
    **kwargs
):
    """Function that updates the first and the second primal variables of ADMM
    based on total variation regularization.

    Args:
        psfs (PSFs): PSFs object containing psfs and other csbs state data
        primal1 (ndarray): first primal variable of ADMM
        primal2 (ndarray): second primal variable of ADMM
        dual (ndarray): dual variable of ADMM
        psfdfts_h_meas (ndarray): FA^Ty term where F is DFT matrix
        SIG_inv (ndarray): spectrum of (A^TA+nu*W^TW + ...)^{-1}
        nu (float): augmented Lagrangian parameter of ADMM (step size)
        lam (list): regularization parameter of dimension num_sources
        kwargs (dict): keyword arguments of parameters related to the regularizer
    Returns:
        ndarray of updated primal1
        ndarray of updated primal2
        ndarray of updated intermediate variable pre_primal2
        ndarray of initialized dual
    """
    [_,num_sources,aa,bb] = psfs.selected_psfs.shape
    ###################  INITIALIZE PRIMAL2 and DUAL ##################
    if primal2 is None:
        primal2 = diff(primal1)
        dual = primal2.copy()

    ######################### PRIMAL1 UPDATE #########################
    pre_primal1 = diff_T(primal2 - dual)
    primal1 = primal1_update_gaussian(SIG_inv=SIG_inv, ATy=psfdfts_h_meas,
        nu=nu, pre_primal1=pre_primal1)
    ######################### PRIMAL2 UPDATE #########################
    pre_primal2 = diff(primal1)
    for i in range(num_sources):
        primal2[:,i] = hard_thresholding(
            pre_primal2[:,i] + dual[:,i],
            threshold = np.sqrt(lam[i]/nu)
        )
    return primal1, primal2, pre_primal2, dual

def bm3d_pnp(
    *,
    psfs,
    primal1,
    primal2,
    dual,
    psfdfts_h_meas,
    SIG_inv,
    nu,
    lam,
    **kwargs
):
    """Function that updates the first and the second primal variables of ADMM
    based on Plug-and-Play regularization with the BM3D denoiser.

    Args:
        psfs (PSFs): PSFs object containing psfs and other csbs state data
        primal1 (ndarray): first primal variable of ADMM
        primal2 (ndarray): second primal variable of ADMM
        dual (ndarray): dual variable of ADMM
        psfdfts_h_meas (ndarray): FA^Ty term where F is DFT matrix
        SIG_inv (ndarray): spectrum of (A^TA+nu*W^TW + ...)^{-1}
        nu (float): augmented Lagrangian parameter of ADMM (step size)
        lam (list): regularization parameter of dimension num_sources
        kwargs (dict): keyword arguments of parameters related to the regularizer
    Returns:
        ndarray of updated primal1
        ndarray of updated primal2
        ndarray of updated intermediate variable pre_primal2
        ndarray of initialized dual
    """
    [_,num_sources,aa,bb] = psfs.selected_psfs.shape
    ###################  INITIALIZE PRIMAL2 and DUAL ##################
    if primal2 is None:
        primal2 = np.zeros_like(primal1)
        dual = primal2.copy()

    ######################### PRIMAL1 UPDATE #########################
    pre_primal1 = primal2 - dual
    primal1 = primal1_update_gaussian(SIG_inv=SIG_inv, ATy=psfdfts_h_meas,
        nu=nu, pre_primal1=pre_primal1)
    ######################### PRIMAL2 UPDATE #########################
    pre_primal2 = primal1
    for i in range(num_sources):
        primal2[i] = pybm3d.bm3d.bm3d(primal1[i]+dual[i], np.sqrt(lam[i]/nu))

    return primal1, primal2, pre_primal2, dual

def dncnn_pnp(
    *,
    psfs,
    primal1,
    primal2,
    dual,
    psfdfts_h_meas,
    SIG_inv,
    nu,
    model,
    **kwargs
):
    """Function that updates the first and the second primal variables of ADMM
    based on Plug-and-Play regularization with the DnCNN denoiser.

    Args:
        psfs (PSFs): PSFs object containing psfs and other csbs state data
        primal1 (ndarray): first primal variable of ADMM
        primal2 (ndarray): second primal variable of ADMM
        dual (ndarray): dual variable of ADMM
        psfdfts_h_meas (ndarray): FA^Ty term where F is DFT matrix
        SIG_inv (ndarray): spectrum of (A^TA+nu*W^TW + ...)^{-1}
        nu (float): augmented Lagrangian parameter of ADMM (step size)
        model (Model): The trained CNN model (Keras)
        kwargs (dict): keyword arguments of parameters related to the regularizer
    Returns:
        ndarray of updated primal1
        ndarray of updated primal2
        ndarray of updated intermediate variable pre_primal2
        ndarray of initialized dual
    """
    [_,num_sources,aa,bb] = psfs.selected_psfs.shape
    ###################  INITIALIZE PRIMAL2 and DUAL ##################
    if primal2 is None:
        primal2 = np.zeros_like(primal1)
        dual = primal2.copy()

    ######################### PRIMAL1 UPDATE #########################
    pre_primal1 = primal2 - dual
    primal1 = primal1_update_gaussian(SIG_inv=SIG_inv, ATy=psfdfts_h_meas,
        nu=nu, pre_primal1=pre_primal1)
    ######################### PRIMAL2 UPDATE #########################
    pre_primal2 = primal1
    for i in range(num_sources):
        noisy = np.reshape(primal1[i]+dual[i], (1,aa,bb,1))
        primal2[i] = model.predict(noisy)[0,:,:,0]

    return primal1, primal2, pre_primal2, dual

def primal1_update_gaussian(
    *,
    pre_primal1,
    SIG_inv,
    ATy,
    nu
):
    """Function that updates the first primal variable of ADMM based on the
    least squares data fidelity term which inherently assumes that the
    measurements have additive Gaussian noise.

    Args:
        pre_primal1 (ndarray): intermediate ADMM variable which is typically
            W(primal2 - dual) where W is the regularization transform
        SIG_inv (ndarray): spectrum of (A^TA+nu*W^TW + ...)^{-1}
        ATy (ndarray): FA^Ty term where F is DFT matrix
        nu (float): augmented Lagrangian parameter of ADMM (step size)
        kwargs (dict): keyword arguments of parameters related to the regularizer
    Returns:
        ndarray of updated primal1
    """
    return np.real(
        np.fft.ifft2(
            block_mul(
                SIG_inv,
                ATy + nu * np.fft.fft2(pre_primal1)
            )
        )
    )

def diff(a):
    """Discrete gradient operator acting horizontally and vertically.

    Periodic boundary condition is assumed at the boundaries.

    Args:
        a (ndarray): 3d array of size (num_sources, aa, bb)

    Returns:
        diff_a (ndarray): 4d array of size (2, num_sources, aa, bb). First
            and second dimensions include the horizontal and vertical gradients,
            respectively.
    """
    [p,aa,bb] = a.shape
    diff_a = np.zeros((2,) + a.shape)
    for i in range(p):
        tempx = a[i].copy()
        tempx[:, 1:] -= a[i, :, :bb-1]
        tempx[:, 0] -= a[i, :, -1]
        diff_a[0, i] = tempx

        tempy = a[i].copy()
        tempy[1:, :] -= a[i, :aa-1, :]
        tempy[0, :] -= a[i, -1, :]
        diff_a[1, i] = tempy

    return diff_a

def diff_T(a):
    """Adjoint of the discrete gradient operator acting horizontally and vertically.

    Periodic boundary condition is assumed at the boundaries.

    Args:
        a (ndarray): 4d array of size (2, num_sources, aa, bb). First and
            second dimensions include the horizontal and vertical gradients,
            respectively.

    Returns:
        ndarray: 3d array of size (num_sources, aa, bb).
    """
    [_,p,aa,bb] = a.shape
    diff_T_a = np.zeros(a.shape)
    for i in range(p):
        tempx = a[0, i].copy()
        tempx[:, :bb-1] -= a[0, i, :, 1:]
        tempx[:, -1] -= a[0, i, :, 0]
        diff_T_a[0, i] = tempx

        tempy = a[1, i].copy()
        tempy[:aa-1, :] -= a[1, i, 1:, :]
        tempy[-1, :] -= a[1, i, 0, :]
        diff_T_a[1, i] = tempy

    return diff_T_a[0] + diff_T_a[1]

def get_SIG_inv(
                *,
                regularizer,
                psfs,
                nu,
                **kwargs
):
    """ The function that returns the spectrum of (A^TA+nu*W^TW + ...)^{-1}

    Args:
        regularizer (function): function that specifies the regularization type
        psfs (PSFs): PSFs object containing psfs and other csbs state data
        kwargs (dict): keyword arguments of parameters

    Returns:
        ndarray of the spectrum of (A^TA+nu*W^TW + ...)^{-1}
    """
    [_,num_sources,aa,bb] = psfs.selected_psfs.shape
    if regularizer.func is TV:
        LAM = get_LAM(rows=aa,cols=bb,order=1)
        spectrum = nu * np.einsum('ij,kl->ijkl', np.eye(num_sources), LAM)

    elif regularizer.func is patch_based:
        psize = np.size(np.empty((6,6,1)))
        LAM = psize * np.ones((aa,bb))
        spectrum = nu * np.einsum('ij,kl->ijkl', np.eye(num_sources), LAM)

    elif regularizer.func is bm3d_pnp:
        LAM = np.ones((aa,bb))
        spectrum = nu * np.einsum('ij,kl->ijkl', np.eye(num_sources), LAM)

    elif regularizer.func is dncnn_pnp:
        LAM = np.ones((aa,bb))
        spectrum = nu * np.einsum('ij,kl->ijkl', np.eye(num_sources), LAM)

    return block_inv(psfs.selected_GAM + spectrum)
