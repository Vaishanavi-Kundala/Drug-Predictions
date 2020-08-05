#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 20:37:34 2020

@author: vaishanavikundala
"""

from sklearn import tree
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
import numpy as np




 #%%

#preprocessing train data
trainX = []
trainY = []

with open ("train_master.txt") as train_file:
    
    for line in train_file:
        line = line.split()
        trainY.append(int(line[0]))
        line = line[1:]
        
        record = [0]*100001
        for data in line:
            data = int(data)
            record[data] = 1
        
        trainX.append(record)
        
    train_file.close()

 #%%  
 
# TruncatedSVD
   
#   find the optimal numer of components
components = [300,400,500,600,700,800,900]
stats = []
for x in components:
    tsvd = TruncatedSVD(n_components= x)
    tsvd.fit(trainX)
    stats.append(tsvd.explained_variance_ratio_.sum())
    print("Number of components = %r and explained variance = %r"%(x,tsvd.explained_variance_ratio_.sum()))
    
plt.plot(components, stats)
plt.xlabel('Number of components')
plt.ylabel("Explained Variance")
plt.title("Plot of Number of components v/s explained variance")
plt.show()


  
tsvd = TruncatedSVD(n_components= 800)
tsvd = tsvd.fit(trainX)
trainX = tsvd.transform(trainX)



 #%%     

# decision tree: cross validation

    X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size= 0.20)

#    Over sample the active compounds
    oversample = SMOTE(k_neighbors = 3)
    X_train, y_train = oversample.fit_resample(X_train, y_train)
    
    dt = tree.DecisionTreeClassifier(class_weight = 'balanced')
    dt = dt.fit(X_train, y_train) 
    predY = dt.predict(X_test) 
    
    print(predY)
    print(confusion_matrix(y_test, predY))
    print(classification_report(y_test, predY))


  






