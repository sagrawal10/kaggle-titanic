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
    
    