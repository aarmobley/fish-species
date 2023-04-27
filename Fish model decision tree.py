# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 21:05:47 2023

@author: Aaron Mobley
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

#create decision tree model
#Trying to accuraely predict fish species based on size

#Import data

fish = pd.read_csv(r"C:\Users\Aaron Mobley\Downloads\archive (4)\Fish.csv")

fish.head()

#x an y data 
x = fish[['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']]
y = fish['Species']

#x and y train test split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=76)

decisionTree = DecisionTreeClassifier(random_state=76)
decisionTree.fit(x_train, y_train)


treePredictions = decisionTree.predict(x_test)

print(confusion_matrix(y_test, treePredictions))
print(classification_report(y_test, treePredictions))

#weighted average accuracy is 84%

#Find outliers, if any
sns.boxplot(x=fish['Length3'])
sns.boxplot(x=fish['Length1'])
sns.boxplot(x=fish['Length2'])
sns.boxplot(x=fish['Weight'])
sns.boxplot(x=fish['Width'])
sns.boxplot(x=fish['Height'])

#Hyperparameter tuning for a decision tree model

param_grid = {
    'max_depth': [2, 4, 6, 8, 10],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 2, 3, 4],
    'criterion': ['gini', 'entropy']}
    
# create decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)
grid_search.fit(x_train, y_train)

print("Best hyperparameters: ", grid_search.best_params_)
print("Best mean cross-validated score: {:.2f}".format(grid_search.best_score_))

#apply new hyperparameters to model
clf = DecisionTreeClassifier(max_depth=8, criterion='gini', min_samples_leaf=1, min_samples_split=6)
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)

print(accuracy_score(y_test, predictions))


#Finding feature importance

feature_importances = pd.Series(clf.feature_importances_, index=x.columns)
feature_importances

#Height is the important feature in determining the species of fish

#Plot showing feature importance = Height
feature_importances.plot(kind='barh', figsize=(7,6))

#correlation matrix 
fish.corr()
corr_ = fish.corr(method='pearson')
pd.DataFrame(corr_).style.background_gradient(cmap='coolwarm')


print(plot)

streamlit run your_script.py
