#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Ensemble Model Prediction of titanic_survived_condition
=========================================================
This example uses datasets titanic.txt from URL:http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt
Then, we use the decision tree model to predict the survival of the Titanic passengers

"""
# Author: HangJie

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

# Feature Selection: select some feature with large relevance related to our classifier task
X = titanic[['pclass','age','sex']]
y = titanic['survived']

# Explore the characteristic of the current selection
print(X.info())

# Data Processing
# Age information completion
X['age'].fillna(X['age'].mean(),inplace=True)

print(X.info())
# Data segmentation
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=33)

# Feature extraction
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient = 'record'))
X_test = vec.transform(X_test.to_dict(orient = 'record'))

# Initialize Decision Tree Classifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_predict = dtc.predict(X_test)

print('the prediction accuracy of DT for titanic:',dtc.score(X_test,y_test))
print(classification_report(y_test,y_predict,target_names=['died','survived']))

# Initialize RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
y_predict = rfc.predict(X_test)

print('the prediction accuracy of RFC for titanic:',rfc.score(X_test,y_test))
print(classification_report(y_test,y_predict,target_names=['died','survived']))

# Initialize GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
y_predict = gbc.predict(X_test)

print('the prediction accuracy of GBC for titanic:',gbc.score(X_test,y_test))
print(classification_report(y_test,y_predict,target_names=['died','survived']))



