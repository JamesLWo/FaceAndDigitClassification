# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    "*** YOUR CODE HERE ***"
    priorProbability = util.Counter() #find proportion of times each label appears in training
    priorCounts = util.Counter()

    for label in trainingLabels:
      priorCounts[label] = priorCounts[label] + 1
      #print("current prior count for label " + str(label) + " is: " + str(priorCounts[label]))
    sizeOfTraining = len(trainingLabels)
    #print("the size of training is " + str(sizeOfTraining))
    for label in self.legalLabels:
      priorProbability[label] = float(priorCounts[label])/sizeOfTraining #priorProbability now has proportion for each label
      #
      # print("the prior probability for label " + str(label) + " is: " + str(priorProbability[label]))
    
    self.priorProbability = priorProbability #save this info

    #now we calculate the evidence: what is the probability of the evidence given the label?

    i = 0
    featuresCounts = util.Counter()
    featuresProbability = util.Counter()

    for datum in trainingData: #for each picture/counter
      label = trainingLabels[i]
      i = i + 1
      for feature in datum: #for each feature in the counter
        #featuresCounts[(feature,label)] += datum[feature]
        if datum[feature] == 1: #if feature is a 1
          featuresCounts[(feature,label)] +=  1 #featuresCounter will keep track of how many pictures in the dataset have 1 for each (feature,label)
          #if x have 1 for feature 1, sizeOfTraining-x have 0 for feature 1
    
    for feature,label in featuresCounts:
      if(featuresCounts[(feature,label)] == 0):
        print("I see a zero")
        exit()
      #print("The feature counts for: " + str(feature) + " and " + str(label) + " is " + str(featuresCounts[(feature,label)]))
    

    for feature,label in featuresCounts:
      featuresProbability[(feature,label)] = float(featuresCounts[(feature,label)]) / priorCounts[label]
      #features probability is the proportion of items where its feature x = 1 (given feature and label)
      #print(str(featuresProbability[(feature,label)]))

    #print("the features probability is: " + str(featuresCounts[(36,41),0]))
    self.featuresProbability = featuresProbability #save this info
    
    #util.raiseNotDefined()
        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum): #datum is the features array 
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    logJoint = util.Counter()
    for label in self.legalLabels:
      totalProbability = 0
      #print("The prior probability: " + str(self.priorProbability[label]))
      totalProbability += math.log(self.priorProbability[label])
      for feature in datum:
        if datum[feature] == 1:
          #print("The featuresProbability for feature " + str(feature) + " and label " + str(label) + " is:  " + str(self.featuresProbability[(feature,label)]))
          if(0 < self.featuresProbability[(feature,label)] and  self.featuresProbability[(feature,label)]< 1):
            totalProbability += math.log(self.featuresProbability[(feature,label)])
          elif(self.featuresProbability[(feature,label)] == 1):
            totalProbability += 0
          else:
            totalProbability -= 1
        else:
          #print("The featuresProbability for feature " + str(feature) + " and label " + str(label) + " is:  " + str(self.featuresProbability[(feature,label)]))
          if(0 < self.featuresProbability[(feature,label)] and  self.featuresProbability[(feature,label)]< 1):
            #print("the value is " + str(1 -self.featuresProbability[(feature,label)]))
            totalProbability += math.log(1 - self.featuresProbability[(feature,label)])
          elif(self.featuresProbability[(feature,label)] == 0):
            totalProbability += 0
          else:
            totalProbability -= 1
      logJoint[label] = totalProbability

    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
  
    
    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []
       
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresOdds
    

    
      
