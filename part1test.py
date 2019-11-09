# Useful unit tests for the methods used by KNN class
# Author: Massimiliano Natale

from part1 import KNN
import numpy as np
import unittest
import mock


class TestKNN(unittest.TestCase):

    def setUp(self):
        np.genfromtxt = mock.MagicMock(return_value=np.empty([2,2]))
        self.target = KNN("", "")

    def test_distinct(self):
        input = [1, 2, 3, 2, 2, 1, 5, 4, 4, 5, 4, 3, 2, 2, 1, 2, 5, 4, 6, 5, 4, 4, 3, 2, 3, 4, 3, 1, 5, 2, 6, 4, 6]
        expected = [1, 2, 3, 5, 4, 6]

        self.assertEqual(self.target._distinct(input), expected)

    def test_calculateDistance(self):
        instance1 = np.array([3, 104])
        instance2 = np.array([18, 90])

        self.assertEqual(self.target.calculateDistance(instance1, instance2), 20.518284528683193)

if __name__=="__main__":
    unittest.main()
