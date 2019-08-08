import numpy as np
from scipy.ndimage import map_coordinates
from scipy.ndimage.interpolation import shift
from scipy.optimize import curve_fit, minimize
from abel.tools.polar import polar2cart, cart2polar, index_coords

def reproject_image_into_polar(data, origin=None, Jacobian=False,
                               dr=1, dt=None, log=False):
    """
    Reprojects a 2D numpy array (``data``) into a polar coordinate system.
    "origin" is a tuple of (x0, y0) relative to the bottom-left image corner,
    and defaults to the center of the image.

    Parameters
    ----------
    data : 2D np.array
    origin : tuple
        The coordinate of the image center, relative to bottom-left
    Jacobian : boolean
        Include ``r`` intensity scaling in the coordinate transform.
        This should be included to account for the changing pixel size that
        occurs during the transform.
    dr : float
        Radial coordinate spacing for the grid interpolation
        tests show that there is not much point in going below 0.5
    dt : float
        Angular coordinate spacing (in radians)
        if ``dt=None``, dt will be set such that the number of theta values
        is equal to the maximum value between the height or the width of
        the image.

    Returns
    -------
    output : 2D np.array
        The polar image (r, theta)
    r_grid : 2D np.array
        meshgrid of radial coordinates
    theta_grid : 2D np.array
        meshgrid of theta coordinates

    Notes
    -----
    Adapted from:
    http://stackoverflow.com/questions/3798333/image-information-along-a-polar-coordinate-system

    """
    # bottom-left coordinate system requires numpy image to be np.flipud
    data = np.flipud(data)

    ny, nx = data.shape[:2]
    if origin is None:
        origin = (nx//2, ny//2)

    # Determine that the min and max r and theta coords will be...
    x, y = index_coords(data, origin=origin)  # (x,y) coordinates of each pixel
    r, theta = cart2polar(x, y)  # convert (x,y) -> (r,θ), note θ=0 is vertical

    nr = np.int(np.ceil((r.max()-r.min())/dr))

    if dt is None:
        nt = max(nx, ny)
    else:
        # dt in radians
        nt = np.int(np.ceil((theta.max()-theta.min())/dt))

    # Make a regular (in polar space) grid based on the min and max r & theta
    if log:
        r_i = np.logspace(np.log10(r.min() + 1), np.log10(r.max()), nr, endpoint=False)
    else:
        r_i = np.linspace(r.min(), r.max(), nr, endpoint=False)
    theta_i = np.linspace(theta.min(), theta.max(), nt, endpoint=False)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Project the r and theta grid back into pixel coordinates
    X, Y = polar2cart(r_grid, theta_grid)

    X += origin[0]  # We need to shift the origin
    Y += origin[1]  # back to the bottom-left corner...
    xi, yi = X.flatten(), Y.flatten()
    coords = np.vstack((yi, xi))  # (map_coordinates requires a 2xn array)

    zi = map_coordinates(data, coords)
    output = zi.reshape((nr, nt))

    if Jacobian:
        output = output*r_i[:, np.newaxis]

    return output, r_grid, theta_grid

def fast_prony2d(y):
    """Fast 2D Prony's method for a single exponential

    Args:
        (ndarray): 2D input array, should be a 2D complex exponential

    Returns:
        (ndarray): length 2 array containing exponential frequency estimate (in Hz, assuming f_s=1)
    """

    def zeropad(x, padded_size):
        """zeropad 1D array x to size padded_size"""

        return np.pad(
            x,
            [(0, padded_size - x.shape[0]), (0, padded_size - x.shape[1])],
            mode='constant'
        )

    y_hat_m = y[:-1, :]
    b_hat_m = -y[1:, :]
    y_hat_n = y[:, :-1]
    b_hat_n = -y[:, 1:]

    padded_size_m = len(y_hat_m) + len(b_hat_m) - 1
    h_hat_m = np.fft.ifft2(
        np.fft.fft2(zeropad(b_hat_m, padded_size_m)) / np.fft.fft2(zeropad(y_hat_m, padded_size_m))
    )
    h_m = [1, h_hat_m[0, 0]]

    padded_size_n = len(y_hat_n) + len(b_hat_n) - 1
    h_hat_n = np.fft.ifft2(
        np.fft.fft2(zeropad(b_hat_n, padded_size_n)) / np.fft.fft2(zeropad(y_hat_n, padded_size_n))
    )
    h_n = [1, h_hat_n[0, 0]]

    omega_m_reconstructed = np.log(np.roots(h_m)) / (1j * 2 * np.pi / 100)
    omega_n_reconstructed = np.log(np.roots(h_n)) / (1j * 2 * np.pi / 100)

    return (omega_m_reconstructed, omega_n_reconstructed)

def phase_correlate(x, y):
    """Perform normalized phase-correlation"""

    return np.fft.ifftn(
        np.fft.fftn(x) * np.fft.fftn(y).conj() /
        np.abs(np.fft.fftn(x) * np.fft.fftn(y))
    )
