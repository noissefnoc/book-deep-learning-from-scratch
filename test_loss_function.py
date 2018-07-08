import unittest
import numpy as np
import loss_function


class TestLossFunction(unittest.TestCase):
    """
    test class of loss function
    """
    # answer is "2"
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    # expect is "2"
    y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    # expect is "7"
    y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.5, 0.0]


    def test_mean_squared_error(self):
        """

        :return:
        """
        self.assertLess(
            loss_function.mean_squared_error(np.array(self.y1), np.array(self.t)),
            loss_function.mean_squared_error(np.array(self.y2), np.array(self.t)))


    def test_cross_entropy_error(self):
        """

        :return:
        """
        self.assertLess(
            loss_function.cross_entropy_error(np.array(self.y1), np.array(self.t)),
            loss_function.cross_entropy_error(np.array(self.y2), np.array(self.t)))