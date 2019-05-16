import numpy as np

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

    assert x.shape[1] == y.shape[0] and x.shape[-2:] == y.shape[-2:], "Matrix dimensions do not agree"
    return np.einsum('ijkl,j...kl->i...kl', x, y)


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
