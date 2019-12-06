import pandas as pd
from numpy import roll
from numpy.fft import fft2, ifft2
from scipy.ndimage import fourier_shift

def experiment(func, iterations, **kwargs):
    """
    Run `func` repeatedly and save the results in a Pandas DataFrame
    Args:
        func (function): function which returns a dict of results
           use `return dict(**locals())` to return all locals in function
        iterations (int): number of iterations to repeat experiment
    """

    result = []
    for t in range(iterations):
        print('Trial {}/{}\r'.format(t + 1, iterations), end='')
        result.append(func(**kwargs))

    return pd.DataFrame(result)


def shift(image, amount, real=True):
    """
    Fourier shift in spatial domain

    Args:
        image: input image
        amount: tuple to shift by
        real (bool): cast image to real (default True)
    """

    result = ifft2(fourier_shift(fft2(image), amount, axis=0))

    return result.real if real else result
