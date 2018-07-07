import unittest
import perceptron

class TestPerceptron(unittest.TestCase):
    """
    test class of perceptron
    """

    def test_AND(self):
        """
        test AND method result
        :return:
        """
        self.assertEqual(perceptron.AND(0, 0), 0)
        self.assertEqual(perceptron.AND(1, 0), 0)
        self.assertEqual(perceptron.AND(0, 1), 0)
        self.assertEqual(perceptron.AND(1, 1), 1)

