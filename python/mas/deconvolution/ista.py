from mas.decorators import vectorize
from mas.forward_model import get_measurements
from skimage.transform import radon, iradon
import numpy as np
import pywt
import sys
from mas.forward_model import size_equalizer

@vectorize
def radon_forward(x,):
    theta = np.linspace(-30., 30., x.shape[0], endpoint=False)
    return radon(x, theta=theta, circle=False)

@vectorize
def radon_adjoint(x):
    theta = np.linspace(-30., 30., x.shape[1], endpoint=False)
    return iradon(x, theta=theta, circle=False, filter=None)

def default_adjoint(x, psfs):
    [p, aa, bb] = x.shape
    [k, p, ss, ss] = psfs.psfs.shape
    ta, tb = [aa + ss - 1, bb + ss - 1]


    # FIXME: make it work for 2D input, remove selected_psfs
    # FIXME: ;move psf_dft computation to PSFs (make PSFs accept sampling_interval and o
    # output size arguments)

    # reshape psfs
    expanded_psfs = size_equalizer(psfs.psfs, ref_size=[aa,bb])

    expanded_psfs = np.repeat(expanded_psfs, psfs.copies.astype(int), axis=0)
    expanded_psf_dfts = np.fft.fft2(expanded_psfs).transpose((1, 0, 2, 3))

    # ----- forward -----
    im = np.fft.fftshift(
        np.fft.ifft2(
            np.einsum(
                'ijkl,jkl->ikl',
                expanded_psf_dfts,
                np.fft.fft2(x)
            )
        ),
        axes=(1, 2)
    )
    # im = get_measurements(sources=x, psfs=psfs, real=True)
    return radon_forward(im)

def default_forward(x, psfs):
    im = radon_adjoint(x)
    return get_measurements(sources=im, psfs=psfs, real=True)


def ista(*, measurements, psfs, lam=10**-5.854, time_step=10**-1.621, iterations=100,
         forward=default_forward, adjoint=default_adjoint, final=radon_adjoint,
         rescale=False, liveplot=False, plt=None):
    """ISTA for arbitrary forward/adjoint transform

    Args:
        forward (function): transformation from sparse -> image domain
        adjoint (function): transformation from image -> sparse domain
        final (function): transformation from sparse -> image domain w/out blur
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
        sys.stdout.write('\033[K')
        print(f'ISTA iteration {n}/{iterations}\r', end='')

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

    result = final(x)
    result -= np.min(result)
    result /= np.max(result)

    return result
