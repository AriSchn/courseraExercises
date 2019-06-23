# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 19:58:11 2019

@author: arschneider
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def computeCost(theta,features,Y):
    J = 0;
    for i in range(len(features)):
        #compute the value of the hypothesis
        h = theta[0] + theta[1]*X[i,1]
        J = J + 1/(2*len(features))*((h-Y[i]))**2
    return J;

def updateTheta(theta,X,alpha):
    sum_0 = 0
    sum_1 = 0
    for i in range(len(X)):
        h = theta[0] + theta[1]*X[i,1]
        dif =  h - Y[i]
        sum_0 = sum_0 + dif*X[i,0]
        sum_1 = sum_1 + dif*X[i,1]
    theta[0] = theta[0] - (alpha*1/(len(data)))*sum_0
    theta[1] = theta[1] - (alpha*1/(len(data)))*sum_1
    return theta
    


data = pd.read_csv('ex1data1.txt',header=None)




data.columns= ['Population','Profit']

print(data.describe())


#Features
X = np.ones(shape = (len(data),1))
features = np.array([data['Population'].values])
features = np.transpose(features)
X = np.append(X,features,axis = 1)

#Target

Y = np.array([data['Profit'].values])
Y = np.transpose(Y)
#initialization of weights
theta = [0,0]

#alpha definition
alpha= 0.01

cost_behavior = np.zeros(1500)
iterations = np.zeros(1500)
for i in range(1500):

   cost =  computeCost(theta,X,Y)
   theta = updateTheta(theta,X,alpha)
   cost_behavior[i] = cost
   iterations[i]= i
#compute two points in the function that was estimated, so we can draw the line
point_1 = theta[0] + theta[1]*5
point_2 = theta[0] + theta[1]*24

x_plot = [5,24]
y_plot = [point_1,point_2]

#draw the line
plt.figure(0)
plt.plot(x_plot,y_plot,'r')

#draw the points of the dataset
plt.scatter(data['Population'],data['Profit'])

plt.figure(1)
plt.plot(iterations[0:100],cost_behavior[0:100])


