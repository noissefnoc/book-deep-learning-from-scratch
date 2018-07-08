#!/usr/bin/env python


def numerical_diff(f, x):
    """

    :param f:
    :param x:
    :return:
    """
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)
