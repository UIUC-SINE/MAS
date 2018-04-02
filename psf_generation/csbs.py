#!/usr/bin/env python3
# Evan Widloski - 2018-04-02
# Clustered SBS Implementation

import numpy as np
import h5py
import timeit

# @profile
def random_cost(measurements):
    """
    Calculate cost from input PSFs

    Args:
        measurements (ndarray): structured numpy array containing 'num_copies' and 'psfs'
    """
    return np.random.rand()

# @profile
def csbs(measurements, cost_func, iterations, **kwargs):
    r"""
    Perform clustered sbs algorithm on a set of psfs

    Args:
        measurements (ndarray): structured numpy array containing 'num_copies' and 'pfs'
        cost_func (function): accepts `measurements` and \**kwargs, and returns scalar cost
        iterations (int): run clustered sbs this many times
        \*kwargs: keyword arguments to pass to cost_func

    Returns:
        measurements with 'num_copies' modified
    """

    for i in range(iterations):
        lowest_psf_group_index = None
        lowest_psf_group_cost = float('inf')
        # iterate psf_group combinations and find the lowest cost
        for psf_group_index in range(len(measurements['psfs'])):
            # print(i, psf_group_index)
            # only evaluate groups with nonzero copies
            if measurements['num_copies'][psf_group_index] >= 1:
                measurements_temp = measurements.copy()
                measurements_temp['num_copies'][psf_group_index] -= 1
                psf_group_cost = cost_func(measurements_temp, **kwargs)
                if psf_group_cost < lowest_psf_group_cost:
                    lowest_psf_group_cost = psf_group_cost
                    lowest_psf_group_index = psf_group_index

        measurements['num_copies'][lowest_psf_group_index] -= 1
        # print(measurements['num_copies'])

    return measurements

# @profile
def main(data_file='/tmp/out.hdf5', cost_func=random_cost, num_copies=10,
         iterations=1):
    """
    Read datafile, build measurements array, and start CSBS algorithm

    Args:
        data_file (str): path to hdf5 datafile containing incoherent PSFs
        cost_func (function): cost function to use during CSBS
        num_copies (int): number of copies of each psf group to initialize algorithm with
        iterations (int): number of iterations of algorithm (number of psf groups removed)
    """

    # load psfs from file and set copies
    with h5py.File(data_file) as f:
        num_images, image_width, image_height = f['incoherentPsf']['value'].shape

        # measurements at each plane
        #  measurements['copies'] number of measurements at this distance
        #  measurements['psfs'] psfs of each source at this distance
        measurements = np.zeros(num_images, dtype=[('num_copies', 'i'),
                                                   ('psfs', 'f', (image_width, image_height))])
        measurements['psfs'] = f['incoherentPsf']['value']

    measurements['num_copies'][:] = num_copies

    return csbs(measurements, cost_func, iterations)

if __name__ == '__main__':
    main()
