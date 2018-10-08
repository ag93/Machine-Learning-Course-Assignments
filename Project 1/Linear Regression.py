# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 17:23:56 2018

@author: Aniket Gade
"""

from sklearn.datasets import load_iris
import numpy as np
#-------------------------------------------------------------------------------------------------------
def linear_regression(X_train, y_train, X_test, y_test):
    
    """ Calculate Beta_hat = (X.T X)^-1 (X.T Y) 
        Where X is the Training Data Set
              Y is the Training Label (Target) Data Set
    """
    lhs = np.dot(np.transpose(X_train), X_train)
    lhs_inverse = np.linalg.inv(lhs)
    rhs = np.matmul(np.transpose(X_train), y_train)
    beta_hat = np.matmul(lhs_inverse, rhs)
    
    """ Predicting The Test Set using beta """
    pred = np.matmul(X_test, beta_hat)
    pred = abs(np.rint(pred)).astype(int)
    print("\nActual Labels -", y_test)
    print("Predicted Labels - ", pred)
    """ Calculating Accuracy of the trained Model """
    count = 0
    for i in range(len(pred)):
        if(pred[i] == y_test[i]):
            count+=1
    accuracy = count/len(pred)
    print('Accuracy = ', accuracy)
    return accuracy
#-------------------------------------------------------------------------------------------------------

def kfold_cv(k, iris_X, iris_y):
    np.random.seed(147)
    indices = np.random.permutation(len(iris_X))    #Creating a Random Permutation for shuffeling the data
    n = len(iris.data)
    len_k = n // k
    accuracy_list = []
    """
    Splitting the Data Into Training and Testing data sets in rotation to perform k-fold CV    
    """
    for i in range(k):
        start = i * len_k
        end = ((i + 1) * len_k)
        iris_X_test  = iris_X[indices[start:end]]
        iris_y_test  = iris_y[indices[start:end]]
        iris_X_train = iris_X[indices[[x for x in indices if x not in indices[start:end]]]]
        iris_y_train = iris_y[indices[[x for x in indices if x not in indices[start:end]]]]
        accuracy = linear_regression(iris_X_train, iris_y_train, iris_X_test, iris_y_test)
        accuracy_list.append(accuracy)
    average_accuracy = sum(accuracy_list)/len(accuracy_list)
    print("\nAverage Accuracy for", k, "fold CV :", average_accuracy)
#---------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    iris = load_iris()
    k_fold = 5
    kfold_cv(k_fold, iris.data, iris.target)

