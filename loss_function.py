#!/usr/bin/env python

import numpy as np


def mean_squared_error(y, t):
    """

    :param y:
    :param t:
    :return:
    """
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    """

    :param y:
    :param t:
    :return:
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    h = t.argmax(axis=1)
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
