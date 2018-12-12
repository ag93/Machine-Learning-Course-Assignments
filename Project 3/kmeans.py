# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 18:22:51 2018

@author: Aniket Gade
"""

import random
from math import floor
import collections

def read_data():
    iris = []
    with open('iris.csv') as f:
        for line in f:
            data = line[:-1].split(',')
            iris.append(data)
    random.shuffle(iris)
    n = len(iris)
    train = floor(n * 0.8)  #Split the data: 80% Clustering and 20% for evaluating
    test = n - train
    return iris, train, test

def kmeans(iris, train, test):
    kmean = random.sample(iris[:train], 10)
    flag = True
    n = len(iris)
    while flag:
        kcluster = [[], [], []]
        next_centroid = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        for data in range(train):   #Iterate throught training data
            dist = [0, 0, 0]
            for k in range(3):      #For 3 Classes
                for i in range(4):  #For 4 Features in each class
                    try:
                        dist[k] = floor(dist[k]) + (abs(float(iris[data][i]) - float(kmean[k][i])) ** 2)    #Calculate Eucledian Distance
                    except ValueError:
                        pass
            min_val = min(dist) #Find Minimum distance of the three classes
            min_index = dist.index(min_val)
            kcluster[min_index].append(iris[data])  #allocate the cluster
            for i in range(4):
                next_centroid[min_index][i] += float(iris[data][i])

        for i in range(3):
            for j in range(4):
                if len(kcluster[i]):
                    next_centroid[i][j] = next_centroid[i][j] / len(kcluster[i])
                else:
                    next_centroid[i][j] = 0.0

        if next_centroid == kmean:  #Algorithm has converged
            flag = False
        else:
            kmean = next_centroid   #New Centroid

    cluster_name = ['', '', '']
    for i in range(3):
        name = []
        for j in range(len(kcluster[i])):
            name.append(kcluster[i][j][4])
        count = collections.Counter(name)
        if not count:
            cluster_name[i] = 'Iris-noise\\n'
        else:
            cluster_name[i] = (count.most_common(1)[0][0])

    #Calculate accuracy
    correct = 0
    for data in range(train, n):
        dist = [0, 0, 0]
        for k in range(3):
            for i in range(4):
                try:
                    dist[k] = floor(dist[k]) + (abs(float(iris[data][i]) - float(kmean[k][i])) ** 2)
                except ValueError:
                    pass
        min_val = min(dist)
        min_index = dist.index(min_val)
        print(iris[data],"=",cluster_name[min_index])
        if iris[data][4] == cluster_name[min_index]:
            correct += 1

    accuracy = correct / test * 100
    print("accuracy =", accuracy)
    print("################################\n")
    return accuracy

if __name__ == "__main__":
    avg_accuracy = 0
    for x in range(5):
        print("Iteration #", x + 1)
        iris, train, test = read_data()
        current_accuracy = kmeans(iris, train, test)
        avg_accuracy += current_accuracy
    
    avg_accuracy = avg_accuracy / 5
    print("average accuracy =", avg_accuracy)
