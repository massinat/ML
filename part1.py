"""
Classification related to part 1.
KNN classification with K=1 and euclidean distance. Votes are not distance weighted.

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
    
    classificationData = knn.buildClassificationData(lambda x: knn.classify(x[:-1], knn._trainingData[:, :-1], 1))
    
    # Save partial result to a file and draw the charts
    resultHelper = ResultHelper("part1.output.txt")

    resultHelper.write(classificationData)
    resultHelper.draw("KNN classification [not-weighted-distance] with K=1")