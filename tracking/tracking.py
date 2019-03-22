from scipy.misc import imread
import numpy as np
from matplotlib import pyplot as plt
import sys

# import sympy as sp

# x = sp.Symbol('x', real=True)

# lam = sp.Piecewise(
#     (0, x < 0),
#     (x, x < 1),
#     (2 - x, x < 2),
#     (0, True)
# )

# sp.plotting.plot(lam, (x, 0, 2))
# TypeError: '<' not supported between instances of 'complex' and 'int'




def convolution_matrix(x, N=None, mode='full'):
    # Author: Jake VanderPlas
    # LICENSE: MIT
    """Compute the Convolution Matrix

    This function computes a convolution matrix that encodes
    the computation equivalent to ``numpy.convolve(x, y, mode)``

    Parameters
    ----------
    x : array_like
        One-dimensional input array
    N : integer (optional)
        Size of the array to be convolved. Default is len(x).
    mode : {'full', 'valid', 'same'}, optional
        The type of convolution to perform. Default is 'full'.
        See ``np.convolve`` documentation for details.

    Returns
    -------
    C : ndarray
        Matrix operator encoding the convolution. The matrix is of shape
        [Nout x N], where Nout depends on ``mode`` and the size of ``x``. 

    Example
    -------
    >>> x = np.random.rand(10)
    >>> y = np.random.rand(20)
    >>> xy = np.convolve(x, y, mode='full')
    >>> C = convolution_matrix(x, len(y), mode='full')
    >>> np.allclose(xy, np.dot(C, y))
    True

    See Also
    --------
    numpy.convolve : direct convolution operation
    scipy.signal.fftconvolve : direct convolution via the
                               fast Fourier transform
    scipy.linalg.toeplitz : construct the Toeplitz matrix
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x should be 1-dimensional")

    M = len(x)
    N = M if N is None else N

    if mode == 'full':
        Nout = M + N - 1
        offset = 0
    elif mode == 'valid':
        Nout = max(M, N) - min(M, N) + 1
        offset = min(M, N) - 1
    elif mode == 'same':
        Nout = max(N, M)
        offset = (min(N, M) - 1) // 2
    else:
        raise ValueError("mode='{0}' not recognized".format(mode))

    xpad = np.hstack([x, np.zeros(Nout)])
    n = np.arange(Nout)[:, np.newaxis]
    m = np.arange(N)
    return xpad[n - m + offset]

image_width = 4096
ccd_width = image_width // 4
pixels_count = 1024
pixel_width = ccd_width // pixels_count
x = imread('sun.jpg')[image_width // 2, :, 0]
# x = np.ones(4096) * 0
# x[1500:2000] = 100

conv = convolution_matrix(x, N=ccd_width)

pixel_photons = []
for pixel in range(pixels_count):
    lam = np.sum(conv[:, np.arange(pixel * pixel_width, pixel * pixel_width + pixel_width)])
    pixel_photons.append(np.random.poisson(lam=lam))

intensity_scaling = sum(pixel_photons) / 10e4
photons = []
for pixel in range(pixels_count):
    print('pixel #{}'.format(pixel))
    lams = np.sum(conv[:, np.arange(pixel * pixel_width, pixel * pixel_width + pixel_width)], axis=1)
#     pixel_photons_count = 0
#     while pixel_photons_count < pixel_photons[pixel] // intensity_scaling:
#         photon_time = np.random.randint(0, len(lams))
#         photon_intensity = np.random.uniform(0, max(lams))
#         if photon_intensity < lams[photon_time]:
#             photons.append((photon_time, pixel))
#             pixel_photons_count += 1

# photons = np.array(photons)
# photons[:, 0] = photons[:, 0] * ccd_width / (image_width + ccd_width - 1)
# middle_point = (512, 512)
# photons = photons - middle_point
# np.save('photons.npy', photons)
# sys.exit()
# photons = np.load('square.npy')
photons = np.load('sun.npy')


xlim = (0, pixels_count - 1)
ylim = (0, len(lams) - 1)



# photons = photons[np.linalg.norm(photons, axis=1) < 512]

# samples = int(10e3)
ylim = (0, 1023)
xlim = (0, 5118)
# photons = np.random.uniform((xlim[0], ylim[0]), (xlim[1], ylim[1]), (samples, 2))

plt.close()
plt.figure()
plt.plot(x)
plt.figure()
plt.subplot(3, 1, 1)
# plt.scatter(photons[1600:3600, 1], photons[1600:3600, 0], s=1)
plt.scatter(photons[:, 1], photons[:, 0], s=1)
plt.ylim([1600, 3600])
plt.xlabel("Pixel")
plt.ylabel("Time step")
plt.axis('equal')

def f(x, theta, xlim, ylim):
    x_0 = ylim[1] * np.sin(theta)
    x_1 = xlim[1] * np.cos(theta)
    if theta < np.tan(ylim[1] / xlim[1]):
        y = ylim[1] / np.sin(np.pi / 2 - theta)
        if theta == 0:
            return y * np.ones(len(x))
        else:
            return (
            (y / x_0) * x
            - (y / x_0) * (x > x_0) * (x - x_0)
            - (y / x_0) * (x > x_1) * (x - x_1)
            )
    else:
        y = xlim[1] / np.sin(theta)
        if theta == np.pi / 2:
            return y * np.ones(len(x))
        else:
            return (
            (y / x_1) * x
            - (y / x_1) * (x > x_1) * (x - x_1)
            - (y / x_1) * (x > x_0) * (x - x_0)
            )

num_bins = 200
hists = np.empty((0, num_bins))
bins = np.empty((0, num_bins))
projecteds = []
scores = []
thetas = np.linspace(1e-6 + np.pi / 100, np.pi - np.pi / 100, 20)
for theta in thetas:
    # photons_masked = photons[
    #     np.logical_and(
    #         np.abs(photons @ np.array([np.cos(theta), np.sin(theta)])) < 305 * np.sqrt(2) / 2,
    #         np.abs(photons @ np.array([np.cos(theta + np.pi / 2), np.sin(theta + np.pi / 2)])) < 305 * np.sqrt(2) / 2,

    #     )
    # ]
    photons_masked = photons[np.linalg.norm(photons, axis=1) < 305]
    projected = photons_masked @ np.array([[np.cos(theta)], [np.sin(theta)]])
    projected = projected.reshape(-1)
    projecteds.append(projected)
    count, bin_edges = np.histogram(projected, bins=num_bins)
    hists = np.append(hists, count[np.newaxis, :], axis=0)
    bins = np.append(bins, bin_edges[:-1][np.newaxis, :], axis=0)
    sorted = np.sort(projected)
    # score = np.sum((f((sorted[:-1] + sorted[1:]) / 2, theta, xlim, ylim) * np.diff(sorted))**2)
    score = np.sum((np.diff(sorted))**2)
    scores.append(score)

    plt.subplot(3, 1, 1)
    # plt.scatter(photons_masked[:, 1], photons_masked[:, 0])
    plt.subplot(3, 1, 2)
    plt.cla()
    plt.gca().set_aspect('equal')
    plt.plot(count)
    plt.subplot(3, 1, 3, projection='polar')
    plt.polar(theta, score, 'o')
    plt.show()
    input()

# plt.scatter(photons_diamond[:, 1], photons_diamond[:, 0])
# plt.scatter(projecteds[:, 1], projecteds[:, 0])
# plt.plot(hists[10, :])
# plt.hist(projected, bins=100)
plt.show()

