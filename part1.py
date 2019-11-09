"""
K-Nearest Neighbor implementation.
@Author: Massimiliano Natale
"""

import math
import operator
import numpy as np
from resultHelper import ResultHelper

class KNN:

    def __init__(self, trainingDataPath, testDataPath):
        self._trainingData = np.genfromtxt(trainingDataPath, delimiter=",")
        self._testData = np.genfromtxt(testDataPath, delimiter=",")
        self._classes = self._distinct(self._trainingData[:, -1])

    """
    Returns an array of distinct elements
    """
    def _distinct(self, enumerable):
        result = []
        for item in enumerable:
            if not item in result:
                result.append(item)
        
        return result

    """
    Helper method to calculate the euclidean distance between 2 distances.
    Returns the euclidean distance between instance1 and instance2.
    """
    def calculateDistance(self, instance1, instance2):
        return math.sqrt(np.sum(np.square(instance1 - instance2)))

    """
    Calculate all the distances between a query instances and the instances of the base truth.
    Returns a tuple with all the calculated distances and the sorted indexes array.
    """
    def calculateDistances(self, queryInstance, trainingInstances):
        distances = [self.calculateDistance(queryInstance, x) for x in trainingInstances]

        return (distances, np.argsort(distances))

    """
    Classify a query instance by using the instances of the base truth.
    It uses a vote done by the nearest k instances.
    Return the winning class, the first winning class if multiple classes get the same vote.
    """
    def classify(self, queryInstances, trainingInstances, k):
        votes = {}

        for item in self._classes:
            votes[item] = 0
        
        sortedIndexes = self.calculateDistances(queryInstances, trainingInstances)[1]
        for i in range(0, k):
            votes[self._trainingData[sortedIndexes[i], -1]] += 1

        return max(votes.items(), key=operator.itemgetter(1))[0]

"""
Trigger the classification.
Accuracy of the model will be calculated based on the partial values written on part1.output.txt.
Charts provided to better understand the behaviour of the algorithm.
"""
if __name__=="__main__":
    knn = KNN("data/classification/trainingData.csv", "data/classification/testData.csv")
    
    numberTotal = 0
    numberCorrects = 0
    numberWrongs = 0
    classificationData = []

    for item in knn._testData:
        numberTotal += 1
        correctClassification = item[-1]
        currentClassification = knn.classify(item[:-1], knn._trainingData[:, :-1], 1)

        if currentClassification==correctClassification:
            numberCorrects += 1
        else:
            numberWrongs += 1

        classificationData.append(f"{numberTotal},{numberCorrects},{numberWrongs}\n")
    
    # Save partial result to a file and draw the charts
    resultHelper = ResultHelper("part1.output.txt")

    resultHelper.write(classificationData)
    resultHelper.draw("Model behaviour K=1")


         



