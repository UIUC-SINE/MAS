from mas.decorators import vectorize
from skimage.transform import radon, iradon
import numpy as np
import pywt

@vectorize
def radon_forward(x, psfs):
    theta = np.linspace(-30., 30., x.shape[0], endpoint=False)
    return radon(x, theta=theta, circle=False)

@vectorize
def radon_adjoint(x, psfs):
    theta = np.linspace(-30., 30., x.shape[1], endpoint=False)
    return iradon(x, theta=theta, circle=False, filter=None)

def ista(*, measurements, psfs, forward=radon_adjoint, adjoint=radon_forward,
         lam=0.025, time_step=0.0005, iterations=100,
         rescale=False, liveplot=False, plt=None):
    """ISTA for arbitrary forward/adjoint transform

    Args:
        forward (function): transformation from sparse -> image domain
        adjoint (function): transformation from image -> sparse domain
        lam (float): soft-threshold value
        time_step (float): gradient step size
        iterations (int): total number of iterations
        scale (bool): rescale image to [0, 1] each iteration
        liveplot (bool): show reconstruction on `plt` every 10 iterations
        plt (figure): figure to use if `liveplot` is True

    Returns:
        ndarray: image reconstruction
        """

    x = adjoint(measurements, psfs)

    for n in range(iterations):
        # print(f'iteration {n}')

        im = forward(x, psfs)

        if liveplot and n % 10 == 0:
            plt.subplot(1, 3, 3)
            plt.imshow(im[0])
            # plt.subplot(2, 3, 6)
            # plt.imshow(im[1, 0])
            plt.show()
            plt.pause(.05)

        if rescale:
            im -= np.min(im)
            im /= np.max(im)

        x = pywt.threshold(
            x + time_step * adjoint(measurements - im, psfs),
            lam
        )

    return forward(x, psfs)
