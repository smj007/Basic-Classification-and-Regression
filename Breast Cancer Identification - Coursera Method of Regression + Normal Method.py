# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 06:17:20 2020

@author: saimi
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv") 
  
print (data.head)

data.info()

data = data.drop(['Unnamed: 32', 'id'], axis = 1)
data.diagnosis = [1 if each == 'M' else 0 for each in data.diagnosis]

y = data.diagnosis.values
x_data = data.drop(['diagnosis'], axis = 1)

x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data)).values


# =============================================================================
# Using hard-coded Logistic Regression - Coursera Method 
# =============================================================================


from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split( 
    x, y, test_size = 0.15, random_state = 42) 

x_train = x_train.T 
x_test = x_test.T 
y_train = y_train.T 
y_test = y_test.T 
  
print("x train: ", x_train.shape) 
print("x test: ", x_test.shape) 
print("y train: ", y_train.shape) 
print("y test: ", y_test.shape) 

def initialize_weights_and_bias(dimension): 
    w = np.full((dimension, 1), 0.01) 
    b = 0.0
    return w, b 

# z = np.dot(w.T, x_train)+b 
def sigmoid(z): 
    y_head = 1/(1 + np.exp(-z)) 
    return y_head 

def forward_backward_propagation(w, b, x_train, y_train): 
    z = np.dot(w.T, x_train) + b 
    y_head = sigmoid(z) 
    loss = - y_train * np.log(y_head) - (1 - y_train) * np.log(1 - y_head) 
    # x_train.shape[1]  is for scaling 
    cost = (np.sum(loss)) / x_train.shape[1]       
  
    # backward propagation 
    derivative_weight = (np.dot(x_train, ( 
        (y_head - y_train).T))) / x_train.shape[1]  
    derivative_bias = np.sum( 
        y_head-y_train) / x_train.shape[1]                  
    gradients = {"derivative_weight": derivative_weight, 
                 "derivative_bias": derivative_bias} 
    return cost, gradients 

def update(w, b, x_train, y_train, learning_rate, number_of_iterations): 
    cost_list = [] 
    cost_list2 = [] 
    index = [] 
  
    # updating(learning) parameters is number_of_iterations times 
    for i in range(number_of_iterations): 
        # make forward and backward propagation and find cost and gradients 
        cost, gradients = forward_backward_propagation(w, b, x_train, y_train) 
        cost_list.append(cost) 
  
        # lets update 
        w = w - learning_rate * gradients["derivative_weight"] 
        b = b - learning_rate * gradients["derivative_bias"] 
        if i % 10 == 0: 
            cost_list2.append(cost) 
            index.append(i) 
            print ("Cost after iteration % i: % f" %(i, cost)) 
  
    # update(learn) parameters weights and bias 
    parameters = {"weight": w, "bias": b} 
    plt.plot(index, cost_list2) 
    plt.xticks(index, rotation ='vertical') 
    plt.xlabel("Number of iterations") 
    plt.ylabel("Cost") 
    plt.show() 
    return parameters, gradients, cost_list 

def predict(w, b, x_test): 
    # x_test is a input for forward propagation 
    z = sigmoid(np.dot(w.T, x_test)+b) 
    Y_prediction = np.zeros((1, x_test.shape[1])) 
  
    # if z is bigger than 0.5, our prediction is sign one (y_head = 1), 
    # if z is smaller than 0.5, our prediction is sign zero (y_head = 0), 
    for i in range(z.shape[1]): 
        if z[0, i]<= 0.5: 
            Y_prediction[0, i] = 0
        else: 
            Y_prediction[0, i] = 1
  
    return Y_prediction 



def logistic_regression(x_train, y_train, x_test, y_test,  
                        learning_rate,  num_iterations): 
  
    dimension = x_train.shape[0] 
    w, b = initialize_weights_and_bias(dimension) 
      
    parameters, gradients, cost_list = update( 
        w, b, x_train, y_train, learning_rate, num_iterations) 
      
    y_prediction_test = predict( 
        parameters["weight"], parameters["bias"], x_test) 
    y_prediction_train = predict( 
        parameters["weight"], parameters["bias"], x_train) 
  
    # train / test Errors 
    print("train accuracy: {} %".format( 
        100 - np.mean(np.abs(y_prediction_train - y_train)) * 100)) 
    print("test accuracy: {} %".format( 
        100 - np.mean(np.abs(y_prediction_test - y_test)) * 100)) 
      
logistic_regression(x_train, y_train, x_test,  
                    y_test, learning_rate = 0.8, num_iterations = 100)



# =============================================================================
# Using imported LR
# =============================================================================
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split( 
    x, y, test_size = 0.15, random_state = 42)

print("x train: ", x_train.shape) 
print("x test: ", x_test.shape) 
print("y train: ", y_train.shape) 
print("y test: ", y_test.shape) 




from sklearn import linear_model 
logreg = linear_model.LogisticRegression() 
logreg.fit(x_train, y_train) 
y_pred = logreg.predict(x_test) 


import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report 
  
cm = confusion_matrix(y_test, y_pred) 
conf_matrix = pd.DataFrame(data = cm,  
                           columns = ['Predicted:0', 'Predicted:1'],  
                           index =['Actual:0', 'Actual:1']) 
plt.figure(figsize = (8, 5)) 
sn.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = "Greens") 
plt.show() 
  
print('The details for confusion matrix is =') 
print (classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)