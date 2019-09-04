import numpy as np

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

            return np.vectorize(func, excluded=excluded, signature=signature)(*args, **kwargs)

        return new_func

    return decorator

vectorize = _vectorize()
