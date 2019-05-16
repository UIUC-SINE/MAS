import numpy as np
from collections import Counter
from scipy.fftpack import dct, dctn, idctn

def patch_extractor(image, *, patch_shape):
    """Create a patch matrix where each column is a vectorized patch.
    It works with both 2D and 3D patches. Patches at the boundaries are
    extrapolated as if the image is periodically replicated. This way all
    the patches have the same dimension.

    Args:
        image (ndarray): 4d array of spectral images
        patch_shape (tuple): tuple of length 3: (dim_x, dim_y, dim_z)

    Returns:
        patch_mtx (ndarray): patch matrix containing vectorized patches in its
        columns. Number of columns (patches) is equal to the number of pixels

        (or voxels) in the image.
    """
    assert len(patch_shape) == 3, 'patch_shape must have length=3'

    [p,_,aa,bb] = image.shape
    patch_size = np.size(np.empty(patch_shape))
    patch_mtx = np.zeros((patch_size, np.size(image)))

    # periodically extend the input image
    temp = np.concatenate((image, image[:patch_shape[2] - 1,:,:,:]), axis = 0)
    temp = np.concatenate((temp, temp[:,:,:patch_shape[0] - 1,:]), axis = 2)
    temp = np.concatenate((temp, temp[:,:,:,:patch_shape[1] - 1]), axis = 3)
    [rows, cols, slices] = np.unravel_index(
        range(patch_size), patch_shape
    )
    for i in range(patch_size):
        patch_mtx[i,:] = np.reshape(
            temp[
                slices[i] : p + slices[i],
                :,
                rows[i] : aa + rows[i],
                cols[i] : bb + cols[i],
            ],
            -1
        )

    return patch_mtx


def patch_aggregator(patch_mtx, *, patch_shape, image_shape):
    """Implements the adjoint of the patch extractor operator.

    Args:
        patch_mtx (ndarray): patch matrix containing vectorized patches in its
        columns. Number of columns (patches) is equal to the number of pixels
        (or voxels) in the image.
        patch_shape (tuple): tuple of length 3: (dim_x, dim_y, dim_z)
        image_shape (tuple): tuple of length 3: (dim_z, 1, dim_x, dim_y)

    Returns:
        image (ndarray): 3D matrix consisting of the aggregated patches
    """
    temp = np.zeros(
        (image_shape[0]+patch_shape[2] - 1,) +
        (1,) +
        (image_shape[2]+patch_shape[0] - 1,) +
        (image_shape[3]+patch_shape[1] - 1,)
    )

    [rows, cols, slices] = np.unravel_index(
    range(patch_mtx.shape[0]), patch_shape)

    for i in range(patch_mtx.shape[0]):
        temp[
            slices[i] : image_shape[0] + slices[i],
            :,
            rows[i] : image_shape[2] + rows[i],
            cols[i] : image_shape[3] + cols[i]
        ] += np.reshape(
            patch_mtx[i,:], image_shape
        )

    temp[:patch_shape[2] - 1,:,:,:] += temp[image_shape[0]:,:,:,:]
    temp[:,:,:patch_shape[0] - 1,:] += temp[:,:,image_shape[2]:,:]
    temp[:,:,:,:patch_shape[1] - 1] += temp[:,:,:,image_shape[3]:]

    return temp[:image_shape[0],:,:image_shape[2],:image_shape[3]]


def lowrank(i, patches_zeromean, window_size, imsize, threshold, group_size):
    """Form a group of similar patches to `i`th patch, and take the lowrank
    approximation of the group with the specified thresholding type and value.

    Since this function is computationally expensive, the similar patches are
    searched only in a window around the given patch, whose size is specified
    by `window_size`. Euclidean distance is used as the similarity metric.

    Args:
        i (int): index of the patch of interest
        patches_zeromean (ndarray): 2d array of zero mean patches
        window_size (tuple): tuple of size of the window in which the patches are searched
        imsize (tuple): length 2 tuple of size of the reconstructed image
        threshold (tuple): thresholds to be applied on the singular values of the group matrix
        group_size (int): number of patches in the group to be formed

    Returns:
        ndarray: low-rank approximation of the matrix of the grouped patches
        ind (ndarray): array of indices of selected patches that are close to
            the `i`th patch
    """
    # find out which image the index i correspond to (to apply threshold accordingly)
    im = np.int(i / (imsize[0]*imsize[1]))

    # get the indices inside the window
    ind_wind = ind_selector(i, imsize=imsize, window_size=window_size)

    ind = ind_wind[
        np.argsort(
            np.linalg.norm(patches_zeromean[ind_wind] - patches_zeromean[i], axis=1)
        )[:group_size]
    ]
    u, s, v_T = np.linalg.svd(patches_zeromean[ind].T, full_matrices=False)
    return u @ np.diag(hard_thresholding(s, threshold=threshold[im])) @ v_T, ind


def ind_selector(i, *, imsize, window_size):
    """Given the shape of a 2d array and an index on that array, return the
    closest set of indices confined in a rectangular area of `window_size`.

    Args:
        i (int): index of the pixel of interest
        window_size (tuple): tuple of size of the window
        imsize (tuple): length 2 tuple of size of the reconstructed image

    Returns:
        ndarray: 1d array of desired indices
    """
    indo = np.zeros(2, dtype=np.int)
    aa, bb = imsize
    im = np.int(i / (aa*bb))
    i1 = i - im * aa*bb
    ind = np.unravel_index(i1, (aa, bb))
    for j in range(2):
        if ind[j] - window_size[j]/2 < 0:
            indo[j] = 0
        elif ind[j] + window_size[j]/2 > imsize[j]:
            indo[j] = imsize[j] - window_size[j]
        else:
            indo[j] = ind[j] - window_size[j]/2

    indx0 = np.kron(
        np.arange(indo[0], indo[0] + window_size[0]),
        np.ones(window_size[1], dtype=np.int)
    )
    indx1 = np.kron(
        np.ones(window_size[0], dtype=np.int),
        np.arange(indo[1], indo[1] + window_size[1])
    )

    return bb*indx0 + indx1 + im * aa*bb


def indsum(x, y, indices):
    """Add the unordered patches in your patch matrix to the ordered one.

    Input matrix `y` has unordered patches in its columns (there may be repeating
    patches). This function aggragates the patches in `y` on `x` specified by
    the `indices`. The fact that there may be repeating patches made this
    function necessary, otherwise the implementation is one line.

    Args:
        x (ndarray): ordered patch matrix on which the aggregation will occur
        y (ndarray): unordered patch patrix whose patches will be added to `x`
        indices (ndarray): 1d array holding the indices of patches in `y`
            (where they belong to in `x`)
    Returns:
        x (ndarray)
    """
    arg_old = np.arange(len(indices))
    ind_old = indices
    while len(arg_old) > 0:
        ind_new, arg_new = np.unique(ind_old, return_index=True)
        arg_new = arg_old[arg_new]
        x[:, ind_new] += y[:, arg_new]
        arg_old = np.array(list((Counter(arg_old) - Counter(arg_new)).keys()), dtype=np.int)
        ind_old = indices[arg_old]
    return x


def soft_thresholding(x, *, threshold):
    """Element-wise soft thresholding function.

    Args:
        x (ndarray): ndarray of any size
        threshold (float): threshold value of the operation

    Returns:
        y (ndarray): element-wise soft thresholded version of `x`
    """
    y = x.copy()
    y[x > threshold] -= threshold
    y[x < -threshold] += threshold
    y[abs(x) <= threshold] = 0
    return y


def hard_thresholding(x, *, threshold):
    """Element-wise hard thresholding function.

    Args:
        x (ndarray): ndarray of any size
        threshold (float): threshold value of the operation

    Returns:
        y (ndarray): element-wise hard thresholded version of `x`
    """
    return x * (abs(x) > threshold).astype(np.int)


def dctmtx(shape):
    """Return the DCT matrix

    Args:
        shape (tuple): original shape of the flattened array that the DCT matrix
        operates on

    Returns:
        ndarray of the DCT matrix
    """
    def dctmtx1d(size):
        mtx = np.zeros((size,size))
        for i in range(size):
            a = np.zeros(size)
            a[i] = 1
            mtx[:,i] = dct(a, norm='ortho')
        return mtx

    if type(shape) is int:
        return dctmtx1d(shape)

    mtx = dctmtx1d(shape[0])
    for s in shape[1:]:
        mtx = np.kron(dctmtx1d(s), mtx)

    return mtx


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
