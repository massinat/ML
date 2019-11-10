"""
Helper class to visualize nicely the classification results.
@Author: Massimiliano Natale
"""

import os
import csv
import matplotlib.pyplot as plt

class ResultHelper:

    def __init__(self, outputFile):
        self.outputFile = outputFile

    def write(self, experimentData):
        if os.path.exists(self.outputFile):
            os.remove(self.outputFile)
    
        with open(self.outputFile, "a+") as txtFile:
            for item in experimentData:
                txtFile.write(item)

    def draw(self, title):
        x = []
        yCorrect = []
        yWrong = []
        yAccuracy = []

        with open(self.outputFile, "r") as csvFile:
            plots = csv.reader(csvFile, delimiter=",")
            for row in plots:
                x.append(int(row[0]))
                yCorrect.append(float(row[1]))
                yWrong.append(float(row[2]))
                yAccuracy.append(float(row[1]) * 100 / float(row[0]))

        plt.subplot(2, 1, 1)
        plt.plot(x, yCorrect)
        plt.plot(x, yWrong)
        plt.annotate(int(yCorrect[-1]), xy=(x[-1] + 3, yCorrect[-1]))
        plt.annotate(int(yWrong[-1]), xy=(x[-1] + 3, yWrong[-1]))
        plt.xlabel("Total instances")
        plt.ylabel("Classified instances")
        plt.title(title)
        plt.legend(["y = Correct", "y = wrong"], loc="upper left")

        plt.subplot(2, 1, 2)
        plt.plot(x, yAccuracy)
        plt.annotate(f"{yAccuracy[-1]}%", xy=(x[-1] + 3, yAccuracy[-1]))
        plt.xlabel("Total instances")
        plt.ylabel("% Accuracy")

        plt.show()

    def drawRSquared(self, title):
        x = []
        y = []

        with open(self.outputFile, "r") as csvFile:
            plots = csv.reader(csvFile, delimiter=",")
            for row in plots:
                x.append(int(row[0]))
                y.append(float(row[-1]))

        plt.plot(x, y)
        plt.annotate("%.2f" % y[-1] + "%", xy=(x[-1] + 3, y[-1]))
        plt.xlabel("Total instances")
        plt.ylabel("R Squared")
        plt.title(title)

        plt.show()
