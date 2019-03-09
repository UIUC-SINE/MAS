#!/usr/bin/env python3
# Ulas Kamaci - 2018-04-02

import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fft2, fftshift, ifftshift

def block_herm(x):
    """Perform block Hermitian transpose on a matrix.

    Calculates a special 'block Hermitian' transpose in
    dimensions `i` and `j` of a 4D matrix of size (i, j, k, l)

    Args:
        x (ndarray): 4D matrix

    Returns:
        ndarray: output matrix
    """
    return np.conj(np.einsum('ijkl->jikl', x))


def block_mul(x, y):
    """Perform block multiplication on two 4D matrices

    Just 2D matrix multiplication except matrix elements are themselves matrices

    Args:
        x (ndarray): 4D matrix of dimension (i, j, k, l)
        y (ndarray): 4D matrix of dimension (j, m, k, l)

    Returns:
        ndarray: 4D matrix of dimension (i, m, k, l)
    """

    assert x.shape[1] == y.shape[0] and x.shape[2:] == y.shape[2:], "Matrix dimensions do not agree"
    return np.einsum('ijkl,jmkl->imkl', x, y)


def block_inv(x, is_herm=False):
    """Computes inverse of compressed block diagonal matrix

    Input matrix is a "compressed" 4D ndarray, where the last two dimensions
    are 2D matrices which hold the diagonal elements of the blocks

    e.g.  If we have the following block matrix

    | a 0 0 d 0 0 |
    | 0 b 0 0 e 0 |
    | 0 0 c 0 0 f |
    | g 0 0 j 0 0 |
    | 0 h 0 0 k 0 |
    | 0 0 i 0 0 l |

    then the compressed form is

    | [a b c]  [d e f] |
    |                  |
    | [g h i]  [j k l] |

    Args:
        x (ndarray): 4D matrix of dimension (i, i, j, k )

    Returns:
        (ndarray): matrix inverse. dimension (i, i, j, k)
    """

    rows, cols, j, k = x.shape

    assert rows == cols, 'input array must be dimension (i, i, j, k)'

    if rows == 1:
        return 1 / x

    a = x[:rows//2, :cols//2, :, :]
    b = x[:rows//2, -cols//2:, :, :]
    c = x[-rows//2:, :cols//2, :, :]
    d = x[-rows//2:, -cols//2:, :, :]


    # precompute inverse of d for efficiency
    d_inv = block_inv(d, is_herm=is_herm)
    t1 = block_mul(b, d_inv)
    t2 = block_mul(d_inv, c)
    t3 = block_mul(b, t2)
    A_inv = block_inv(a - t3, is_herm=is_herm)

    # https://en.wikipedia.org/wiki/Block_matrix#Block_matrix_inversion
    B_inv = -block_mul(A_inv, t1)
    if not is_herm:
        C_inv = -block_mul(t2, A_inv)
    else:
        C_inv = block_herm(B_inv)
    D_inv = d_inv - block_mul(t2, B_inv)

    return np.concatenate((np.concatenate((A_inv, B_inv), axis=1),
                           np.concatenate((C_inv, D_inv), axis=1)), axis=0)


def block_inv2(x):
    y = np.zeros_like(x)
    for i in range(x.shape[2]):
        for j in range(x.shape[3]):
            t = x[:,:,i,j]
            y[:,:,i,j] = 1/(t[0,0]*t[1,1]-t[1,0]*t[0,1])*np.array([[t[1,1],-t[0,1]],[-t[1,0],t[0,0]]])
    return y

def block_inv3(A):
    from scipy.linalg import lapack
    # lapack_routine = lapack_lite.dgesv
    # Looking one step deeper, we see that solve performs many sanity checks.
    # Stripping these, we have:
    b = np.identity(A.shape[0], dtype=A.dtype)

    identity  = np.eye(A.shape[0])
    def lapack_inverse(a):
        b = np.copy(identity)
        return lapack.dgesv(a, b)[2]

    out = np.zeros_like(A)
    for i in range(A.shape[2]):
        for j in range(A.shape[3]):
            out[:,:,i,j] = lapack_inverse(A[:,:,i,j])
    return out

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
    psf_dfts = np.fft.fft2(psfs.psfs, axes=(2, 3))

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

def get_LAM(*,rows,cols,order):
    """Compute the spectrum of the discrete derivative matrix

    Args:
        rows (int): number of rows in the output
        cols (int): number of cols in the output
        order (int): {0,1,2} order of the derivative matrix

    Returns:
        2d array of the spectrum
    """
    if order is 0:
        return np.ones((rows,cols))
    else:
        diffx_kernel = np.zeros((rows,cols))
        diffy_kernel = np.zeros((rows,cols))
        if order is 1:
            diffx_kernel[0,0] = 1 ; diffx_kernel[0,1] = -1
            diffy_kernel[0,0] = 1 ; diffy_kernel[1,0] = -1
        elif order is 2:
            diffx_kernel[0,0] = 1 ; diffx_kernel[0,1] = -2 ; diffx_kernel[0,2] = 1
            diffy_kernel[0,0] = 1 ; diffy_kernel[1,0] = -2 ; diffx_kernel[2,0] = 1
        return (
            np.abs(np.fft.fft2(diffx_kernel))**2 +
            np.abs(np.fft.fft2(diffy_kernel))**2
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
