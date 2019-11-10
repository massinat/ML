"""
K-Nearest Neighbor implementation.
All the algorithm is implemented in this class, classification can be of 2 types:
 - Not distance weighted classification [triggered by the method classify()]
 - Distance weighted classification [triggered by the method classifyWithDistanceWeight()]

Configuration parameters like k and n can be set as input parameters of these methods.

@Author: Massimiliano Natale
"""

import math
import operator
import numpy as np

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
    It uses a vote not-distance-weighted done by the nearest k instances.
    Return the winning class, the first winning class if multiple classes get the same vote.
    """
    def classify(self, queryInstance, trainingInstances, k):
        votes = {}

        for item in self._classes:
            votes[item] = 0
        
        sortedIndexes = self.calculateDistances(queryInstance, trainingInstances)[1]
        for i in range(0, k):
            votes[self._trainingData[sortedIndexes[i], -1]] += 1

        return max(votes.items(), key=operator.itemgetter(1))[0]

    """
    Classify a query instance by using the instances of the base truth.
    It uses a vote distance-weighted done by the nearest k instances.
    Return the winning class, the first winning class if multiple classes get the same vote.
    """
    def classifyWithDistanceWeight(self, queryInstance, trainingInstances, k, n):
        votes = {}
        for item in self._classes:
            votes[item] = 0

        distanceInfo = self.calculateDistances(queryInstance, trainingInstances)
        distances = distanceInfo[0]
        sortedIndexes = distanceInfo[1]
        for i in range(0, k):
            votes[self._trainingData[sortedIndexes[i], -1]] += math.pow(1 / distances[sortedIndexes[i]], n)

        return max(votes.items(), key=operator.itemgetter(1))[0]

    """
    Calculate the regression for a query instance by using the instances of the base truth.
    It uses a distance-weighted calculation done on the nearest k instances.
    """
    def regressionWithDistanceWeight(self, queryInstance, trainingInstances, k, n):
        regression = 0
        inverseDistance = 0
        sumAllKInverseDistances = 0

        distanceInfo = self.calculateDistances(queryInstance, trainingInstances)
        distances = distanceInfo[0]
        sortedIndexes = distanceInfo[1]
        for i in range(0, k):
            inverseDistance = 1 / math.pow(distances[sortedIndexes[i]], n)
            sumAllKInverseDistances += inverseDistance
            regression += inverseDistance * self._trainingData[sortedIndexes[i], -1]

        return regression / sumAllKInverseDistances

    def buildClassificationData(self, lambdaClassification):
        numberTotal = 0
        numberCorrects = 0
        numberWrongs = 0
        classificationData = []

        for item in self._testData:
            numberTotal += 1
            correctClassification = item[-1]
            currentClassification = lambdaClassification(item)

            if currentClassification==correctClassification:
                numberCorrects += 1
            else:
                numberWrongs += 1

            classificationData.append(f"{numberTotal},{numberCorrects},{numberWrongs}\n")

        return classificationData

    def buildRegressionData(self, lambdaRegression):
        numberTotal = 0
        currentSquaredResiduals = 0
        currentSumOfSquares = 0
        averagePrediction = np.average(self._testData[:, -1])
        regressionData = []

        for item in self._testData:
            numberTotal += 1
            regression = lambdaRegression(item)

            currentSquaredResiduals += math.pow(regression - item[-1], 2)
            currentSumOfSquares += math.pow(averagePrediction - item[-1], 2)
            currentRSquared = 1 - (currentSquaredResiduals / currentSumOfSquares)
            
            regressionData.append(f"{numberTotal},{regression},{item[-1]},{currentRSquared}\n")

        return regressionData