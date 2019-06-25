# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 21:21:33 2019

@author: arschneider
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

def plotdata(data):
    #plot data
    plt.figure()

    positive = data[data['Admitted']==1]
    negative = data[data['Admitted']==0]

    plt.scatter(positive['Score1'].values,positive['Score2'].values, c = 'b', marker = 'o', label = 'Admitted')
    plt.scatter(negative['Score1'].values,negative['Score2'].values, c = 'r', marker = 'x', label = 'Rejected')
    plt.legend()
    plt.xlabel('Exam_1 Score')
    plt.ylabel('Exam_2 Score')
    
def sigmoid(z):
    G = 1/(1 + np.exp(-z))
    G.shape = (len(z),1)
    return G

def costfunction(theta,x,y):
    m = len(x)
    z = x.dot(theta.transpose())
    first = -y * np.log(sigmoid(z))
    second = np.multiply((1-y),np.log((1-sigmoid(z))))
    cost = np.sum(first - second)/m
    return cost

def gradientfunction(theta,x,y):
    m = len(x)
    grad = np.zeros(3)
   
    error = sigmoid(x.dot(theta.transpose())) - y
    
    for i in range(len(grad)):
        feature = np.array([x[:,i]]).transpose()
        term = np.multiply(error,feature)
        grad[i] = np.sum(term)/m
        
    return grad

def plotDecisionBoundary(theta,x):
    index_min = np.where(x[:,1] == np.amin(x[:,1]))
    index_max = np.where(x[:,1] == np.amax(x[:,1]))

    x1 = np.array(x[index_min])
    x2 = np.array(x[index_max])

    x1 = x1[:,1]
    x2 = x2[:,1]
    x_test = np.append(x1,x2,axis=0)
    y_test = -(theta[0] + theta[1]*x_test)/theta[2]
    plt.plot(x_test,y_test,'r')
    
#read the dataset   
data = pd.read_csv('ex2data1.txt',header = None, names = ['Score1','Score2','Admitted'])

#prepare input and output
x = np.array([data['Score1'].values,data['Score2'].values]).transpose()
#Adding the intercept terms to x
ones = np.ones([len(x),1])
x = np.append(ones,x,1)
y = np.array([data['Admitted']]).transpose()
#initialize coeficients
theta = np.zeros(3)
theta.shape = (1,3)

#compute the optimal theta parameters
result = opt.fmin_tnc(func=costfunction, x0=theta, fprime=gradientfunction, args=(x, y))
cost = costfunction(result[0], x, y)

theta = result[0]

plotdata(data)
plotDecisionBoundary(theta,x)


