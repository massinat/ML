"""
Regression related to part 3.
KNN regression with variable K and euclidean distance. Result is distance weighted.

@Author: Massimiliano Natale
"""

from knn import KNN
from resultHelper import ResultHelper

"""
Trigger the regression.
Create the output file and the chart to visualize the result.
"""
if __name__=="__main__":
    knn = KNN("data/regression/trainingData.csv", "data/regression/testData.csv")
    
    regressionData = knn.buildRegressionData(lambda x: knn.regressionWithDistanceWeight(x[:-1], knn._trainingData[:, :-1], 10, 2))
    
    # Save partial result to a file and draw the charts
    resultHelper = ResultHelper("part3.output.txt")

    resultHelper.write(regressionData)
    resultHelper.drawRSquared("KNN regression [weighted-distance] with K=10 and N=2")