import util
import numpy
PRINT = True

class knnClassifier:

    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "knn"
        self.kconstant = 3
    
    def getHeightWidth(self):
        if(len(self.legalLabels) == 2):
            return 60,70
        elif(len(self.legalLabels) == 10):
            return 28,28
    
    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        width, height = self.getHeightWidth()
        training_set = []
        for datum in trainingData:
            image = numpy.zeros((width, height))
            for feature in datum:
                image[feature] = datum[feature]
            training_set.append(image)
        
        self.trainingLabels = trainingLabels
        self.numpyData = training_set

    def classify(self, testData):
        width, height = self.getHeightWidth()
        # guesses is just a list of your answers for each test datum
        guesses = []    

        # create some distance array
        for testDatum in testData:
            distances = []
            testImage = numpy.zeros((width, height))
            for feature in testDatum:
                testImage[feature] = testDatum[feature]
            for trainDatum in self.numpyData:
                distance = numpy.linalg.norm(trainDatum - testImage)
                distances.append(distance)

            sort = sorted(distances)
            ksort = sort[0:self.kconstant]
            closest = []
            for dis in ksort:
                closest.append(self.trainingLabels[distances.index(dis)])
            label = max(set(closest), key=closest.count)                
            
            guesses.append(label)
        
        return guesses