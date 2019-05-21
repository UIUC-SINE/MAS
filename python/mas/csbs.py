#!/usr/bin/env python3
# Evan Widloski - 2018-04-02
# Clustered SBS Implementation

import numpy as np
import timeit
import sys
from mas import psf_generator


def csbs(psfs, cost_module, end_copies, **kwargs):
    r""" Perform clustered sbs algorithm on a set of psfs

    Args:
        psfs (PSFs): PSFs object containing psfs and other csbs state data
        cost_module (module): a python module containing `cost`, and optionally `init` and `iteration_end` functions.
            `init` is called before the CSBS algorithm begins with `psfs` and can
                store initialization data in `psfs` to be used in `cost` and `iteration_end`
            `cost` is called each iteration with `psfs` and the currently removed psf_group `psf_group_index`
            `iteration_end` is called after each iteration with `psfs`, and `lowest_psf_group_index`
                which is the index of the psf group which incurred the lowest cost on the previous iteration
        end_copies (int): run clustered sbs until there are this many copies left
        kwargs: extra keyword arguments to pass to cost_module.cost
    """

    assert end_copies > 0, ("end_copies must be positive")

    # save csbs parameters in PSFs object
    psfs.csbs_params = {
        'end_copies': end_copies,
        'cost_module': cost_module.__name__,
        **kwargs
    }

    # call 'init' if it exists
    if hasattr(cost_module, 'init'):
        cost_module.init(psfs, **kwargs)

    while np.sum(psfs.copies) > end_copies:
        sys.stdout.write('\033[K')
        print('CSBS copies remaining: {}\r'.format(np.sum(psfs.copies).astype(int)), end='')
        lowest_psf_group_index = None
        lowest_psf_group_cost = float('inf')
        # iterate psf_group combinations and find the lowest cost
        costs = []
        for psf_group_index in range(len(psfs.psfs)):
            # only evaluate groups with nonzero copies
            if psfs.copies[psf_group_index] >= 1:
                psf_group_cost = cost_module.cost(
                    psfs,
                    psf_group_index,
                    **kwargs
                )
                costs.append(psf_group_cost)
                # if psf_group_cost < lowest_psf_group_cost:
                if (psf_group_cost < lowest_psf_group_cost) and not np.isclose(psf_group_cost, lowest_psf_group_cost, rtol=0, atol=1e-13):
                    lowest_psf_group_cost = psf_group_cost
                    lowest_psf_group_index = psf_group_index

        # import ipdb; ipdb.set_trace()
        # permanently remove the psf group which incurred the lowest cost
        psfs.copies[lowest_psf_group_index] -= 1
        psfs.copies_history.append(lowest_psf_group_index)

        # call 'iteration_end' if it exists
        if hasattr(cost_module, 'iteration_end'):
            cost_module.iteration_end(psfs, lowest_psf_group_index)
