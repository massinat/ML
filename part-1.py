# K-Nearest Neighbor implementation
# Author: Massimiliano Natale

from numpy import numpy as np
from math import math

class KNN:

    # This is an helper method to calculate the euclidean distance between 2 distances
    def calculateDistance(self, instance1, instance2):
        return math.sqrt(np.sum(2**(instance1 - instance2)))

    # Calculate all the distances between a query instances and the instances of the base truth
    def calculateDistances(self, queryInstance, baseTruthInstances):
        return [self.calculateDistance(queryInstance, x) for x in baseTruthInstances]

