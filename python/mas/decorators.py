import numpy as np

def _vectorize(signature='(m,n)->(i,j)'):
    """Decorator to make a 2D functions work with higher dimensional arrays
    Last 2 dimensions are taken to be images

    Args:
        signature (str): override mapping behavior
    """
    def decorator(func):
        return np.vectorize(
            func,
            # excluded=np.arange(1, func.__code__.co_argcount) ,
            excluded=func.__code__.co_varnames[:func.__code__.co_argcount],
            signature=signature
        )
    return decorator

vectorize = _vectorize()
