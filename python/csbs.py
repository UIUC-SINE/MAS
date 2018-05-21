#!/usr/bin/env python3
# Evan Widloski - 2018-04-02
# Clustered SBS Implementation

import numpy as np
import timeit
import psf_generator

def csbs(measurements, cost_module, iterations, **kwargs):
    r"""
    Perform clustered sbs algorithm on a set of psfs

    Args:
        measurements (ndarray): structured numpy array containing 'num_copies' and 'psfs'
        cost_module (module): a python module containing `init`, `cost`, and `iteration_end` functions.
            `init` is called before the CSBS algorithm begins with `measurements` and can
                return initialization data which will be passed to the `cost` function.
            `cost` is called each iteration with `measurements` and `initialized_data` returned from `init`.
            `iteration_end` is called after 1 iteration and a psf group has been removed with `measurements`,
                `initialized_data`, and `lowest_psf_group_index` which is the index of the psf group which
                incurred the lowest cost on the previous iteration
        iterations (int): run clustered sbs this many times
        kwargs: extra keyword arguments to pass to cost_module.cost

    Returns:
        measurements with 'num_copies' modified
    """
    assert (iterations < np.sum(measurements['num_copies'])), "`iterations` must be less than the total number of psf groups"

    initialized_data = cost_module.init(measurements)
    for i in range(iterations):
        lowest_psf_group_index = None
        lowest_psf_group_cost = float('inf')
        # iterate psf_group combinations and find the lowest cost
        for psf_group_index in range(len(measurements['psfs'])):
            # only evaluate groups with nonzero copies
            if measurements['num_copies'][psf_group_index] >= 1:
                # remove a psf group and check cost
                measurements['num_copies'][psf_group_index] -= 1
                psf_group_cost = cost_module.cost(measurements,
                                                  initialized_data,
                                                  psf_group_index,
                                                  **kwargs)
                if psf_group_cost < lowest_psf_group_cost:
                    lowest_psf_group_cost = psf_group_cost
                    lowest_psf_group_index = psf_group_index
                # add the psf group back
                measurements['num_copies'][psf_group_index] += 1

        # permanently remove the psf group which incurred the lowest cost
        measurements['num_copies'][lowest_psf_group_index] -= 1

        cost_module.iteration_end(measurements, initialized_data, lowest_psf_group_index)

    return measurements

def main():

    measurements = psf_generator.generate_measurements()
    # measurements = psf_generator.load_measurements('/tmp/out.hdf5')

    import random_cost
    iterations = 1
    return csbs(measurements, random_cost, iterations)

if __name__ == '__main__':
    main()
