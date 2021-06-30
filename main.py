# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import csv
import sys
from collections import defaultdict

import numpy
from sklearn import svm
from sklearn import tree
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree, model_selection
import pydot

training_set = pd.read_csv('titanic/train.csv')
test_set = pd.read_csv('titanic/test.csv')

pd.set_option('display.max_columns',20)
pd.set_option('display.width', 1000)

#print (training_set.head())

training_set.drop(columns=['Cabin'], axis=1, inplace=True)
test_set.drop(columns=['Cabin'], axis=1, inplace=True)

training_set['Age'].fillna(training_set['Age'].median(), inplace=True)
training_set['Embarked'].fillna(training_set['Embarked'].mode(), inplace=True)

test_set['Age'].fillna(test_set['Age'].median(), inplace=True)
test_set['Fare'].fillna(test_set['Fare'].median(), inplace=True)

indexes = training_set.iloc[:891,:].index[training_set.iloc[:891,:].SibSp == 8]
training_set.drop(indexes, inplace=True)

indexes = training_set.index[training_set.Parch == 6]
training_set.drop(indexes, inplace=True)


indexes = training_set.index[training_set.Fare > 100]
training_set.drop(indexes, inplace=True)

training_set.drop(columns=['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
test_set.drop(columns=['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

plt.figure(figsize=(16, 14))
sn.set(font_scale=1.2)
sn.set_style('ticks')

training_set = pd.get_dummies(training_set, columns=['Pclass', 'Sex', 'Embarked' ], drop_first= True)
test_set = pd.get_dummies(test_set, columns=['Pclass', 'Sex', 'Embarked' ], drop_first= True)

x_train = training_set.iloc[:,training_set.columns != 'Survived']
y_train = training_set.iloc[:,training_set.columns == 'Survived'].values.reshape(-1,1)

x_test = test_set

#print(x_test.isna().sum())

classifier_dt = DecisionTreeClassifier(random_state=1, max_depth=7, min_samples_leaf=5)
classifier_dt.fit(x_train, y_train)

print(classifier_dt.score(x_train, y_train))

scores = model_selection.cross_val_score(classifier_dt, x_train, y_train, scoring="accuracy", cv=50)
print(scores)
print(scores.mean())

tree.export_graphviz(classifier_dt, feature_names=x_train.columns,out_file="tree.dot")

#classifier_xgb = XGBClassifier(use_label_encoder=True)
#classifier_xgb.fit(x_train, y_train)
y_pred_xgb = classifier_dt.predict(x_test)

test_set = pd.read_csv('titanic/test.csv')

output = pd.DataFrame({'PassengerId': test_set.PassengerId, 'Survived': y_pred_xgb})

output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")