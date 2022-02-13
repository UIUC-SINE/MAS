import pandas as pd
import numpy as np
from itertools import product
import functools, operator
from tqdm import tqdm

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

    print('\033[K', end='')
    return pd.DataFrame(result)

def xy2rc(x, y=None):
    if y is None:
        x, y = x.T

    return np.array((-y, x)).T

def rc2xy(r, c=None):
    if c is None:
        r, c = r.T

    return np.array((c, -r)).T

def combination_experiment(func, disable_print=False, iterations=1, **kwargs):
    """
    Run `func` with all combinations of input parameters and return results in
    dataframe

    Args:
        func (function): function which returns a dict of results
            use `return dict(**locals())` to return all function variables
        iterations (int): number of iterations to repeat each experiment
        disable_print (boolean): disable tqdm printing
        kwargs: keyword arguments that will be passed to `func`.  each kwarg
            must be iterable
    """

      # clear any left over progressbars if in ipython
    # https://github.com/tqdm/tqdm/issues/375#issuecomment-576863223
    getattr(tqdm, '_instances', {}).clear()

    try:
        total = functools.reduce(operator.mul, map(len, kwargs.values()))
        total *= iterations
    except TypeError:
        raise Exception("arguments must be list or iterable")

    results = []
    with tqdm(desc='Trials', total=total, leave=None, disable=disable_print) as tqdm_bar:
        for values in product(*kwargs.values()):
            for _ in range(iterations):
                func_kwargs = dict(zip(kwargs.keys(), values))
                result = func(**func_kwargs)
                tqdm_bar.update(1)

                if type(result) is not dict:
                    result = {'result': result}

                results.append({**result, **func_kwargs})

    return pd.DataFrame(results)


def shift(image, amount, real=True):
    """
    Fourier shift in spatial domain

    Args:
        image: input image
        amount: tuple to shift by
        real (bool): cast image to real (default True)
    """
    from scipy.ndimage import fourier_shift
    from numpy import roll
    from numpy.fft import fft2, ifft2

    result = ifft2(fourier_shift(fft2(image), amount, axis=0))

    return result.real if real else result
