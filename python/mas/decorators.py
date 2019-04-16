import numpy as np

def vectorize(func):
    """Decorator to make a 2D functions work with higher dimensional arrays
    First argument of the function is the higher dimensional array.
    Last 2 dimensions of the array are taken to be images
    """

    return np.vectorize(
        func,
        # ignore all arguments except first positional argument
        excluded=np.arange(1, func.__code__.co_argcount),
        signature='(m,n)->(i,j)'
    )
