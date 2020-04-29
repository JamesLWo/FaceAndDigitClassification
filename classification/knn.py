import util
import numpy as np
PRINT = True

class knnClassifier:

    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "knn"
        self.kConstant = 3
    
    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        if len(self.legalLabels) == 2:
            self.width = 60
            self.height = 70
        else:
            self.width = 28
            self.height = 28
        training_set = []
        for datum in trainingData:
            image = np.zeros((self.height, self.width))
            for feature in datum:
                image[feature[1],feature[0]] = datum[feature]
            training_set.append(image)
            
        self.training_set = training_set
        self.trainingLabels = trainingLabels

    def classify(self, testData):
        testing_set = []
        for datum in testData:
            image = np.zeros((self.height,self.width))
            for feature in datum:
                image[feature[1],feature[0]] = datum[feature]
            testing_set.append(image)

        # guesses is just a list of your answers for each test datum
        guesses = []

        for testDatum in testing_set:
            #for each testDatum, construct a distances list that keeps track of how far away this testDatum is to all training data
            distances = []
            for trainDatum in self.training_set:
                #calculate euclidiean distance between training datum and test datum
                distances.append(np.linalg.norm(testDatum - trainDatum))
            
            #return the indices of the k smallest distances
            indicesList = sorted(range(len(distances)), key = lambda sub: distances[sub])[:self.kConstant]

            #Go through each index in the indicesList to find the k corresponding labels for the k closest training data
            trainingLabelOccurrences = util.Counter()
            for i in indicesList:
                trainingLabelOccurrences[self.trainingLabels[i]] += 1


            mostFrequentLabel = 0
            frequency = 0
            for label in trainingLabelOccurrences:
                if trainingLabelOccurrences[label] > frequency:
                    mostFrequentLabel = label
                    frequency = trainingLabelOccurrences[label]

            #append the label that occurred most often
            guesses.append(mostFrequentLabel)

        return guesses