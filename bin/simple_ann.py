# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 10:33:01 2019

@author: Shreya
"""

#Data Preprocessing

import numpy as np
import pandas as pd

def steps(file_name):
    #Importing Data
    df = pd.read_csv(file_name)
    
    #Data Preprocessing
    df.Age = df.Age.fillna(df.Age.median())
    df.Embarked = df.Embarked.fillna('S')
    
    # Encoding categorical data
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'])
    
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    if 'Survived' in df.columns:
        y = df.loc[:, 'Survived'].values
        df = df.drop('Survived', axis=1)
    else:
        y = None
    
    X = df.values
    
    return X, y
    


# Kaggle Competition
# Shreya Agrawal | 22Dec2018
# Using cross validation and regularisation (Dropout)

# Part 1 - Data Preprocessing

import numpy as np
import pandas as pd

#Training Data
X , y = steps("../data/train.csv")

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part2: Constructing the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(6, kernel_initializer = "uniform", activation='relu', input_shape=(10,)))
    classifier.add(Dropout(rate=0.1)) #Disable 10% of neurons in each iteration
    classifier.add(Dense(6,  kernel_initializer = "uniform", activation = 'relu'))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(1, kernel_initializer = "uniform", activation = 'sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])    
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv=10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()

classifier.fit(X_train, Y_train)

test = pd.read_csv('../data/test.csv')
test_Survived = pd.Series(classifier.predict(test).flat, name="Survived")

results = pd.concat([IDtest,test_Survived],axis=1)

results.to_csv("../data/simple_ann.csv",index=False)