import numpy as np
from functools import wraps
import inspect


def _vectorize(signature='(m,n)->(i,j)', included=[0]):
    """Decorator to make a 2D functions work with higher dimensional arrays
    Last 2 dimensions are taken to be images
    Iterate over first position argument.

    Args:
        signature (str): override mapping behavior
        included (list): list of ints and strs of position/keyword arguments to iterate over

    Returns:
        function: decorator which can be applied to nonvectorized functions

    Signature examples:

        signature='(),()->()', included=[0, 1]
            first two arguments to function are vectors. Loop through each
            element pair in these two vectors and put the result in a vector of
            the same size. e.g. if args 0 and 1 are of size (5, 5), the output
            from the decorated function will be a vector of size (5, 5)

        signature='(m,n)->(i,j)', included=[0]
            input argument is a vector with at least two dimensions. Loop
            through all 2d vectors (using last 2 dimensions) in this input
            vector, process each, and return a 2d vector for each. e.g. if arg
            0 is a vector of size (10, 5, 5), loop through each (5, 5) vector
            and return a (10, 5, 5) vector of all the results

        signature='(m,n)->()', included=[0]
            input argument is a vector with at least two dimensions. Loop
            through each 2d image and return a 1d vector. e.g. if arg 0 is a
            vector of size (10, 5, 5), return a vector of size (10)
    """
    def decorator(func):

        def new_func(*args, **kwargs):
            nonlocal signature

            # exclude everything except included
            excluded = set(range(len(args))).union(set(kwargs.keys()))
            excluded -= set(included)

            # allow signature override
            if 'signature' in kwargs.keys():
                signature = kwargs['signature']
                kwargs.pop('signature')

            # print(f'{func=}')
            # print(f'{excluded=}')
            # print(f'{signature=}')
            return np.vectorize(func, excluded=excluded, signature=signature)(*args, **kwargs)

        return new_func

    return decorator

vectorize = _vectorize()

def store_kwargs(func):
    """
    Apply to any class __init__ to automatically store all kwargs inside the class
    https://stackoverflow.com/questions/1389180/automatically-initialize-instance-variables
    """
    names, varargs, keywords, defaults = inspect.getargspec(func)

    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper

def np_gpu(np_args=[], np_kwargs=[]):
    """
    Apply to functions to use cupy in place of numpy if available.
    function must have a `np` kwarg and should return a single numpy array

    Args:
        np_args (list): list of positional arguments to convert from np.ndarray
            to cupy.ndarray
        np_kwargs (list): list of keyword arguments to convert from np.ndarray
            to cupy.ndarray

    """

    def decorator(func):

        try:
            import cupy

            @wraps(func)
            def wrapper(*args, **kwargs):
                args = list(args)
                for np_arg in np_args:
                    args[np_arg] = cupy.array(args[np_arg])
                for np_kwarg in np_kwargs:
                    kwargs[np_kwarg] = cupy.array(kwargs[np_kwarg])

                result = func(*args, np=cupy, **kwargs)
                if type(result) is cupy.ndarray:
                    return cupy.asnumpy(result)
                else:
                    return result

        except ModuleNotFoundError:
            import numpy as np

            @wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, np=np, **kwargs)
                return result

        return wrapper

    return decorator
