# Kaggle Competition
# Shreya Agrawal | 22Dec2018

#Data Preprocessing

import numpy as np
import pandas as pd
import preProcessing as pp

#Training Data
X , y = pp.steps("../data/train.csv")

#df = pd.read_csv('final_train.csv')
#y = df.Survived
#X = df.drop('Survived', axis=1)

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
from keras.layers import Dense


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    #Adding input layer and the first hidden layer
    classifier.add(Dense(5, kernel_initializer = "uniform", activation='relu', input_shape=(10,)))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(5,  kernel_initializer = "uniform", activation = 'relu'))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(1, kernel_initializer = "uniform", activation = 'sigmoid'))
    #Compiling the ANN
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv=10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()


# Fit it to the training set
classifier.fit(X_train, y_train, batch_size=10, epochs=50)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

#Getting the accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

#Getting the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


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
df_output = {"PassengerId":df.PassengerId.values.tolist(), "Survived": y_real.flat}
df_out = pd.DataFrame(df_output)
df_out.to_csv('../data/simple_ann.csv', index=False)


# This gives a result of 0.77990