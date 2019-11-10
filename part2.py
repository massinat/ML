"""
Classification related to part 2.
KNN classification with variable K and euclidean distance. Votes are distance wighted.

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
    
    numberTotal = 0
    numberCorrects = 0
    numberWrongs = 0
    classificationData = []

    for item in knn._testData:
        numberTotal += 1
        correctClassification = item[-1]
        currentClassification = knn.classifyWithDistanceWeight(item[:-1], knn._trainingData[:, :-1], 10, 2)

        if currentClassification==correctClassification:
            numberCorrects += 1
        else:
            numberWrongs += 1

        classificationData.append(f"{numberTotal},{numberCorrects},{numberWrongs}\n")
    
    # Save partial result to a file and draw the charts
    resultHelper = ResultHelper("part2.output.txt")

    resultHelper.write(classificationData)
    resultHelper.draw("Model behaviour K=10 distance weighted")