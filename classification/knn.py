import util
import numpy as np
PRINT = True

class knnClassifier:

    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "knn"
    
    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        training_set = []
        for datum in trainingData:
            image = numpy.zeroes((width, height))
            for feature in datum:
                image[feature] = datum[feature]
            training_set.append(image)
            
        self.numpyData = training_set

    def classify(self, testData):
        # guesses is just a list of your answers for each test datum
        guesses = []

        # create some distance array
        for testDatum in testData:
            for trainDatum in self.numpyData:
                #calculate distance between training datum and test datum
                #insert distance into distance array
                #get like
                pass 
            guesses.append(answer)
        
        return guesses