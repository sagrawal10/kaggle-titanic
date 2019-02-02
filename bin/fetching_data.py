# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 09:17:30 2019

@author: Shreya
"""

# Ensemble Learning in Python
# Titanic Kaggle Problem

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

sns.set(style='white', context='notebook', palette='deep')

#Load Data
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
IDtest = test["PassengerId"]



#Outlier Detection
"""
Takes a pandas dataframe and  returns a list of the indices corresponsing 
to the observations containing more than n outliers according to the Tukey method.
"""
def detect_outliers(df,n,features):
    outlier_indices = []
    # iterate over features(columns)
    for col in features:
        Q1 = np.percentile(df[col], 25) # 1st quartile (25%)
        Q3 = np.percentile(df[col],75) # 3rd quartile (75%)
        IQR = Q3 - Q1   # Interquartile range (IQR)
        
        outlier_step = 1.5 * IQR    # outlier step
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
    
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    return multiple_outliers   
    
# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])
train = train.drop(Outliers_to_drop, axis=0).reset_index(drop=True)



#joinng train and test data for categorical conversion
train_len = len(train)
dataset = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

#Fill empty and NaNs values with NaN
dataset = dataset.fillna(np.nan)
#dataset.isnull().sum()     #check for null values, count

import pickle
train.to_pickle('../data/train.pkl')
dataset.to_pickle('../data/dataset.pkl')

#Feature Analysis: Based on Graphical Analysis. Refer to graphical_analysis.ipynb

