import util
PRINT = True

class knnClassifier:

    def __init__(self, legalLabels, neighbors):
        self.legalLabels = legalLabels
        self.type = "knn"
        self.neighbors = neighbors
    
    def classify(self, data, trainingData, trainingLabels):
        