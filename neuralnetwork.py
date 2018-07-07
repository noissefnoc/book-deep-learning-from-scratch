#!/usr/bin/env python

import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    """

    :param x:
    :return:
    """
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    """

    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1)

    # for step funciton
    y = step_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()

    y1 = sigmoid(x)
    plt.plot(x, y1)
    plt.ylim(-0.1, 1.1)
    plt.show()
