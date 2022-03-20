# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 20:57:16 2022

@author: Srirag Jayakumar
"""
import numpy as np

# To calculate the activation score (Σwixi) using dot product
def activationScore(X,w): 
    a_score = np.inner(X,w)    
    return a_score

# Main perceptron algorithm that updates the weights based on misclassification
def Perceptron(X, Y, lmbda=0):
    w = np.zeros(len(X[0]))
    epochs = 20

    for epoch in range(epochs):
        for i in range(len(X)):
            if (Y[i]*activationScore(X[i], w)) <= 0:
                w = (1-(2*lmbda))*w + (Y[i]*X[i]) # L2 regularization
    return w

# Classifies outputs to +1 class or -1 class based on the activation score
def predictClass(X,w):    
    a_score = activationScore(X,w)    
    return -1 if a_score <=0 else 1

# Calculates the accuracy between the predicted output and true output    
def accuracyScore(X,Y,W):
    predictions=[]
    for val in X:
        predictions.append(predictClass(val,W))
        
    accuracy = (predictions == Y).mean()*100
    return accuracy
    
if __name__ == '__main__':
    # Sample dataset
    X_train = np.array([[5.5,3.3,1.7,0.2],[4.8,3,2.4,0.4],[4.2,3.2,1.5,0.2],[3.6,3.1,1.3,0.3]])
    Y_train = np.array([1,-1,1,-1])
    weights = Perceptron(X_train,Y_train)
    accuracy = accuracyScore(X_train,Y_train,weights)