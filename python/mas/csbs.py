#!/usr/bin/env python3
# Evan Widloski - 2018-04-02
# Clustered SBS Implementation

import numpy as np
import timeit
from mas import psf_generator

def csbs(measurements, cost_module, iterations, **kwargs):
    r""" Perform clustered sbs algorithm on a set of psfs

    Args:
        measurements (Measurements): Measurements object containing psfs and other csbs state data
        cost_module (module): a python module containing `cost`, and optionally `init` and `iteration_end` functions.
            `init` is called before the CSBS algorithm begins with `measurements` and can
                store initialization data in `measurements` to be used in `cost` and `iteration_end`
            `cost` is called each iteration with `measurements` and the currently removed psf_group `psf_group_index`
            `iteration_end` is called after each iteration with `measurements`, and `lowest_psf_group_index`
                which is the index of the psf group which incurred the lowest cost on the previous iteration
        iterations (int): run clustered sbs this many times
        kwargs: extra keyword arguments to pass to cost_module.cost
    """

    assert (iterations < np.sum(measurements.copies)), "`iterations` must be less than the total number of psf groups"

    # save csbs parameters in Measurements object
    measurements.csbs_params = {'iterations': iterations,
                                'cost_module': cost_module.__name__,
                                **kwargs}

    # call 'init' if it exists
    if hasattr(cost_module, 'init'):
        cost_module.init(measurements)

    for i in range(iterations):
        lowest_psf_group_index = None
        lowest_psf_group_cost = float('inf')
        # iterate psf_group combinations and find the lowest cost
        for psf_group_index in range(len(measurements.psfs)):
            # only evaluate groups with nonzero copies
            if measurements.copies[psf_group_index] >= 1:
                # remove a psf group and check cost
                measurements.copies[psf_group_index] -= 1
                psf_group_cost = cost_module.cost(measurements,
                                                  psf_group_index,
                                                  **kwargs)
                if psf_group_cost < lowest_psf_group_cost:
                    lowest_psf_group_cost = psf_group_cost
                    lowest_psf_group_index = psf_group_index
                # add the psf group back
                measurements.copies[psf_group_index] += 1

        # permanently remove the psf group which incurred the lowest cost
        measurements.copies[lowest_psf_group_index] -= 1
        measurements.copies_history.append(lowest_psf_group_index)

        # call 'iteration_end' if it exists
        if hasattr(cost_module, 'iteration_end'):
            cost_module.iteration_end(measurements, lowest_psf_group_index)

def main():

    measurements = psf_generator.generate_measurements()

    import random_cost
    iterations = 1
    csbs(measurements, random_cost, iterations)

    return measurements

if __name__ == '__main__':
    main()
