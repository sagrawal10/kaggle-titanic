# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 07:59:25 2019

@author: Shreya
"""

# Kaggle Competition
# Shreya Agrawal | 22Dec2018
# Using cross validation and regularisation (Dropout)

# Part 1 - Data Preprocessing

import numpy as np
import pandas as pd


train = pd.read_pickle('../data/train_final.pkl')
dataset = pd.read_pickle('../data/dataset_final.pkl')

from fetching_data import train_len
train = dataset[:train_len]
test = dataset[train_len:]
test.drop(labels=["Survived"],axis = 1,inplace=True)

train["Survived"] = train["Survived"].astype(int)

Y_train = train["Survived"]

X_train = train.drop(labels = ["Survived"],axis = 1)

# Part2: Constructing the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(33, kernel_initializer = "uniform", activation='relu', input_shape=(66,)))
    classifier.add(Dropout(rate=0.1)) #Disable 10% of neurons in each iteration
    classifier.add(Dense(33,  kernel_initializer = "uniform", activation = 'relu'))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(1, kernel_initializer = "uniform", activation = 'sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])    
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv=10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# Prediction
classifier.fit(X_train, Y_train)

test_Survived = pd.Series(classifier.predict(test).flat, name="Survived")

results = pd.concat([IDtest,test_Survived],axis=1)

results.to_csv("../data/modelling_ann.csv",index=False)


# This yielded a score of 0.76076