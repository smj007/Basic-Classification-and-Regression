# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv("weight-height.csv")

dataset.info()
dataset.describe()
dataset.isnull().sum()

dataset['Gender'].replace('Female', 0, inplace = True)
dataset['Gender'].replace('Male', 1, inplace = True)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 2].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

lin_pred = lin_reg.predict(X_test)

import sklearn.metrics
accuracy = sklearn.metrics.r2_score(y_test, lin_pred)


my_weight_pred = lin_reg.predict([[1,65.74]])
