import numpy as np
import pandas as pd
import matplotlib as plt
from collections import Counter

class KNN():

    def euclidean(self,x,y): #euclidean distance 
        return (np.sqrt(np.sum(np.square(x-y))))
    
    def __init__(self,k = 3): #k value 
        self.k = k

    def fit(self, X,Y): #knn model has no complex thing in fit as compared to other model
        self.X_ = X
        self.Y_ = Y

    def predict(self,X):
        prediction = [self._predict(x) for x in X] #calling prediction
        return prediction
    
    def _predict(self,x):
        distance = [self.euclidean(x,y) for y in self.X_] #calculating the eucl dist for each point
        inds = np.argsort(distance)[:self.k] #argsort gives the index of the sorted array, and returns the first k values
        nearest = [self.Y_[i] for i in inds] #gives the labels of the k nearest points
        common = Counter(nearest).most_common() #gives the most common label
        return common[0][0] #returns the most common label

#declaring, fitting, prediction and accuracy checking
# classifier = KNN(5)
# classifier.fit(X_train,Y_train)
# y_pred = classifier.predict(X_test)
# self_acc = np.sum(y_pred==y_test)/len(y_pred)