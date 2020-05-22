# -*- coding: utf-8 -*-
"""
Created on Sun May  3 01:35:29 2020

@author: saimi
"""

# Regression Example With Boston Dataset: Baseline
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# load dataset
from sklearn.datasets import load_boston
boston_dataset = load_boston()

dataset = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
dataset['MEDV'] = boston_dataset.target
# split into input (X) and output (Y) variables
X = dataset.iloc[:,0:13].values
Y = dataset.iloc[:,13].values
# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
# evaluate model
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))

#Previous case - Not efficient - no scaling of data


# Regression Example With Boston Dataset: Standardized
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# load dataset

from sklearn.datasets import load_boston
boston_dataset = load_boston()

dataset = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
dataset['MEDV'] = boston_dataset.target
# split into input (X) and output (Y) variables
X = dataset.iloc[:,0:13].values
Y = dataset.iloc[:,13].values
# define base model
def baseline_model():
    model = Sequential()
    model.add(Dense(13, input_dim = 13, activation = 'relu', kernel_initializer = 'normal' ))
    model.add(Dense(1, kernel_initializer = 'normal'))
    
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    return model
# evaluate model with standardized dataset
estimators = []
estimators.append(("standardize", StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn = baseline_model, epochs = 50, batch_size = 5, verbose = 0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits = 10)
results = cross_val_score(pipeline, X, Y, cv = kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))


#-------Using a DEEPER MODEL--------------

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# load dataset

from sklearn.datasets import load_boston
boston_dataset = load_boston()

dataset = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
dataset['MEDV'] = boston_dataset.target
# split into input (X) and output (Y) variables
X = dataset.iloc[:,0:13].values
Y = dataset.iloc[:,13].values
# define base model
def baseline_model():
    model = Sequential()
    model.add(Dense(13, input_dim = 13, activation = 'relu', kernel_initializer = 'normal' ))
    model.add(Dense(6, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(1, kernel_initializer = 'normal'))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    return model
# evaluate model with standardized dataset
estimators = []
estimators.append(("standardize", StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn = baseline_model, epochs = 50, batch_size = 5, verbose = 0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits = 10)
results = cross_val_score(pipeline, X, Y, cv = kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))


#----------Using a WIDER model----------
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# load dataset

from sklearn.datasets import load_boston
boston_dataset = load_boston()

dataset = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
dataset['MEDV'] = boston_dataset.target
# split into input (X) and output (Y) variables
X = dataset.iloc[:,0:13].values
Y = dataset.iloc[:,13].values
# define base model
def baseline_model():
    model = Sequential()
    model.add(Dense(20, input_dim = 13, activation = 'relu', kernel_initializer = 'normal' ))
    model.add(Dense(1, kernel_initializer = 'normal'))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    return model
# evaluate model with standardized dataset
estimators = []
estimators.append(("standardize", StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn = baseline_model, epochs = 50, batch_size = 5, verbose = 0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits = 10)
results = cross_val_score(pipeline, X, Y, cv = kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))




