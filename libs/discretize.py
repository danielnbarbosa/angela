import numpy as np

def create_uniform_grid(low, high, bins=(10,)):
    """Define a uniformly-spaced grid that can be used to discretize a space.

    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    bins : tuple
        Number of bins along each corresponding dimension.

    Returns
    -------
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    """

    split_list = []
    for dim_low, dim_high, dim_bin in zip(low, high, bins):
        step = (dim_high - dim_low) / dim_bin
        split_list.append(np.linspace(dim_low+step, dim_high-step, num=dim_bin-1))
    return split_list
