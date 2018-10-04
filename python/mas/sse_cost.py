#!/usr/bin/env python3
# Ulas Kamaci - 2018-04-02

import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fft2, fftshift, ifftshift

class block_array(np.ndarray):
    """Subclass of ndarray which redefines multiplication as block_mul
    """
    def __mul__(self, a):
        return block_mul(self, a)
    def __rmul__(self, a):
        return self.__mul__(a)


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
    return np.einsum('ijkl,jmkl->imkl', x, y).view(block_array)


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

    x = x.view(block_array)

    rows, cols, j, k = x.shape

    assert rows == cols, 'input array must be dimension (i, i, j, k)'

    if rows == 1:
        return 1 / x

    a = x[:rows//2, :cols//2, :, :]
    b = x[:rows//2, -cols//2:, :, :]
    c = x[-rows//2:, :cols//2, :, :]
    d = x[-rows//2:, -cols//2:, :, :]


    # precompute inverse of d for efficiency
    d_inv = block_inv(d)
    t1 = b * d_inv
    t2 = d_inv * c
    t3 = b * t2
    A_inv = block_inv(a - t3)


    # https://en.wikipedia.org/wiki/Block_matrix#Block_matrix_inversion
    A = A_inv
    B = -A_inv * t1
    if not is_herm:
        C = -t2 * A_inv
    else:
        C = block_herm(B)
    D = d_inv + t2 * B

    return np.concatenate((np.concatenate((A, B), axis=1),
                           np.concatenate((C, D), axis=1)), axis=0).view(block_array)


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


def init(measurements):
    """
    """
    _, _, rows, cols = measurements.psfs.shape
    # Dx = np.kron(np.eye(rows), diff_matrix(cols))
    # Dy = np.kron(diff_matrix(cols), np.eye(rows))
    psf_dfts = np.fft.fft2(measurements.psfs, axes=(2, 3))

    LAM_idft = np.zeros((rows, cols))
    LAM_idft[0, :] = (diff_matrix(cols).T @ diff_matrix(cols))[0]
    LAM_idft[:, 0] = (diff_matrix(rows).T @ diff_matrix(rows))[0]

    initialized_data = {
        "psf_dfts": psf_dfts,
        "GAM": block_mul(block_herm(psf_dfts), psf_dfts),
        "LAM": np.fft.fft2(LAM_idft)
    }

    measurements.initialized_data = initialized_data


def iteration_end(measurements, lowest_psf_group_index):
    """
    """
    measurements.initialized_data['GAM'] -= block_mul(
        block_herm(measurements.initialized_data['psf_dfts'][lowest_psf_group_index:lowest_psf_group_index + 1]),
        measurements.initialized_data['psf_dfts'][lowest_psf_group_index:lowest_psf_group_index + 1]
    )


def cost(measurements, psf_group_index, **kwargs):
    """
    """

    _, num_sources, _, _ = measurements.psfs.shape

    SIG_e_dft = (
        measurements.initialized_data['GAM'] -
        block_mul(
            block_herm(measurements.initialized_data['psf_dfts'][psf_group_index:psf_group_index + 1]),
            measurements.initialized_data['psf_dfts'][psf_group_index:psf_group_index + 1]
        ) +
        kwargs['lam'] * np.einsum('ij,kl', np.eye(num_sources), measurements.initialized_data['LAM'])
    )

    return np.sum(np.trace(block_inv(SIG_e_dft)))


def main():
    import numpy as np
    from scipy import misc
    import scipy.stats as st
    from matplotlib import pyplot as plt
    from skimage.transform import resize

    def gauss2D(shape=(3,3),sigma=0.5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m,n = [(ss-1.)/2. for ss in shape]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    block_dim = 31
    fft_dim = 51

    '''h = gauss2D(shape=(block_dim,block_dim),sigma=0.5)
    H = block_fft2(h,block_dim,fft_dim)

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(h)
    plt.subplot(122)
    plt.imshow(abs(H))
    plt.show()'''



    x1 = misc.imread('SI1_1.jpg', flatten='true')/255
    x2 = misc.imread('SI1_2.jpg', flatten='true')/255
    print('x1_energy = %d ' % sum(sum(x1*x1)))
    print('x2_energy = %d ' % sum(sum(x2*x2)))

    # fig = plt.figure()
    # plt.subplot(121)
    # plt.imshow(x1, cmap='gray')
    # plt.subplot(122)
    # plt.imshow(x2, cmap='gray')
    # plt.show()

    x = np.concatenate((x1,x2),axis=0)
    fig = plt.figure()
    plt.imshow(x, cmap='gray')
    plt.show()

    imsize = x1.shape[0]
    print('imsize=%d' % imsize)

    h11 = gauss2D(shape=(block_dim,block_dim),sigma=1)
    h12 = gauss2D(shape=(block_dim,block_dim),sigma=20)
    h21 = gauss2D(shape=(block_dim,block_dim),sigma=20)
    h22 = gauss2D(shape=(block_dim,block_dim),sigma=1)

    psf_mtx = np.concatenate((np.concatenate((h11,h12),axis=1),np.concatenate((h21,h22),axis=1)),axis=0)
    otf_mtx = block_fft2(psf_mtx,block_dim,imsize)

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(psf_mtx)
    plt.subplot(122)
    plt.imshow(abs(otf_mtx))
    plt.show()

    Fx = block_fft2(x,imsize,imsize)
    fig = plt.figure()
    plt.imshow(abs(Fx)**0.2, cmap='gray')
    plt.show()

    y = block_ifft2(my_mul(otf_mtx,Fx,imsize),imsize)

    fig = plt.figure()
    plt.imshow(abs(y), cmap='gray')
    plt.show()

    '''numblocks = 100
    block_dim = 128

    a = np.random.random((block_dim*numblocks,block_dim*numblocks))
    ah = my_herm(a,block_dim)
    aha = my_mul(ah,a,block_dim)
    b = my_inv(aha,block_dim)
    c = my_mul(aha,b,block_dim)

    plt.figure()
    plt.imshow(aha)
    plt.show()

    plt.figure()
    plt.imshow(c)
    plt.show()'''

if __name__ == '__main__':
    main()
