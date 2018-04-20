#!/bin/env python
# Cost Module Example
# Evan Widloski - 2018-04-15
import numpy as np

def init(measurements):
    return 'initialized_data'

def cost(measurements, initialized_data):
    """
    Calculate cost from input PSFs

    Args:
        measurements (ndarray): structured numpy array containing 'num_copies' and 'psfs'
    """
    return np.random.rand()
