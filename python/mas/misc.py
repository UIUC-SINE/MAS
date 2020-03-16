import pandas as pd
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

    return pd.DataFrame(result)


def combination_experiment(func, print=False, **kwargs):
    """
    Run `func` with all combinations of input parameters and return results in
    dataframe

    Args:
        func (function): function which returns a dict of results
            use `return dict(**locals())` to return all function variables
        iterations (int): number of iterations to repeat each experiment
        kwargs: keyword arguments that will be passed to `func`.  each kwarg
            must be iterable
    """

    # clear any left over progressbars if in ipython
    # https://github.com/tqdm/tqdm/issues/375#issuecomment-576863223
    getattr(tqdm, '_instances', {}).clear()

    total = functools.reduce(operator.mul, map(len, kwargs.values()))

    results = []
    for values in tqdm(
            product(*kwargs.values()), desc='Trials', total=total, leave=None
    ):

        func_kwargs = dict(zip(kwargs.keys(), values))
        result = func(**func_kwargs)

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
