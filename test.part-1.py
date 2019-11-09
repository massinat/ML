# Useful unit tests for the methods used by KNN class
# Author: Massimiliano Natale

from part1 import KNN
import unittest


class TestKNN(unittest.TestCase):
    def __init__(self):
        self.target = None

    def setUp(self):
        self.target = KNN("data/classification/trainingData.csv", "data/classification/testData.csv")

    def test_distinct(self):
        input = [1, 2, 3, 2, 2, 1, 5, 4, 4, 5, 4, 3, 2, 2, 1, 2, 5, 4, 6, 5, 4, 4, 3, 2, 3, 4, 3, 1, 5, 2, 6, 4, 6]
        expected = [1, 2, 3, 4, 5, 6]

        self.assertEqual(self.target._distinct(input), expected)

if __name__=="__main__":
    unittest.main()
