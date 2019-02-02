# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 03:37:29 2018

@author: Shreya
"""


import numpy as np
import pandas as pd
import preProcessing as pp

#Data Testing
# -----------------------------------------------------------------------------

#Loading the model
from keras.models import load_model
classifier = load_model('model_ann.h5')

#Importing Data
Xt, yt = pp.steps("../data/test.csv")

#df = pd.read_csv('final_test.csv')
#yt = df.Survived
#Xt = df.drop('PassengerId', axis=1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xt = sc.fit_transform(Xt)

y_real = classifier.predict(Xt)
y_real = np.where(y_real[:,0]>0.6, 1, 0)

df = pd.read_csv('../data/test.csv')
df['Survived'] = y_real
df_output = {"PassengerId":df.PassengerId.values.tolist(), "Survived": y_real.tolist()}
df_out = pd.DataFrame(df_output)
df_out.to_csv('foo.csv', index=False)
