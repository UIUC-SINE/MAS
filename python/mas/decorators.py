import numpy as np

def _vectorize(signature='(m,n)->(i,j)', included=[0]):
    """Decorator to make a 2D functions work with higher dimensional arrays
    Last 2 dimensions are taken to be images
    Iterate over first position argument.

    Args:
        signature (str): override mapping behavior
        included (list): list of ints and strs of position/keyword arguments to iterate over
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

            return np.vectorize(func, excluded=excluded, signature=signature)(*args, **kwargs)

        return new_func

    return decorator

vectorize = _vectorize()
