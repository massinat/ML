"""
Classification related to part 2.
KNN classification with variable K and euclidean distance. Votes are distance weighted.

@Author: Massimiliano Natale
"""

from knn import KNN
from resultHelper import ResultHelper

"""
Trigger the classification.
Create the output file and the chart to visualize the result.
"""
if __name__=="__main__":
    knn = KNN("data/classification/trainingData.csv", "data/classification/testData.csv")
    
    #K=10, n=2
    classificationData = knn.buildClassificationData(lambda x: knn.classifyWithDistanceWeight(x[:-1], knn._trainingData[:, :-1], 10, 2))
        
    # Save partial result to a file and draw the charts
    resultHelper = ResultHelper("part2.output.txt")

    resultHelper.write(classificationData)
    resultHelper.draw("KNN classification [weighted-distance] with K=10 and N=2")

    #K=20, n=2
    classificationData = knn.buildClassificationData(lambda x: knn.classifyWithDistanceWeight(x[:-1], knn._trainingData[:, :-1], 20, 2))
        
    # Save partial result to a file and draw the charts
    resultHelper = ResultHelper("part2.output.txt")

    resultHelper.write(classificationData)
    resultHelper.draw("KNN classification [weighted-distance] with K=20 and N=2")

    #K=20, n=4
    classificationData = knn.buildClassificationData(lambda x: knn.classifyWithDistanceWeight(x[:-1], knn._trainingData[:, :-1], 20, 4))
        
    # Save partial result to a file and draw the charts
    resultHelper = ResultHelper("part2.output.txt")

    resultHelper.write(classificationData)
    resultHelper.draw("KNN classification [weighted-distance] with K=20 and N=4")

    #K=30, n=2
    classificationData = knn.buildClassificationData(lambda x: knn.classifyWithDistanceWeight(x[:-1], knn._trainingData[:, :-1], 30, 2))
        
    # Save partial result to a file and draw the charts
    resultHelper = ResultHelper("part2.output.txt")

    resultHelper.write(classificationData)
    resultHelper.draw("KNN classification [weighted-distance] with K=30 and N=2")