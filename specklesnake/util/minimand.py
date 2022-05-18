import numpy as np


def zero_normalized_sum_of_square_difference(array_f, array_g):
    """ Evaluate the zero normalized sum of square differences between two
    arrays

    Parameters
    ----------
    array_f : ndarray
        2D array representing first array
    array_g : ndarray
        2D array representing second array

    Returns
    -------
    znssd : float
        Zero normalized sum of square difference
    """
    array_f_bar = (array_f - array_f.mean()) / array_f.std()
    array_g_bar = (array_g - array_g.mean()) / array_g.std()
    return np.sum(np.square(array_f_bar) - np.square(array_g_bar))
