# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:09:25 2019

@author: arschneider
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt


def plotdata(data):
    #plot data
    plt.figure()

    positive = data[data['Result']==1]
    negative = data[data['Result']==0]

    plt.scatter(positive['Test1'].values,positive['Test2'].values, c = 'b', marker = 'o', label = 'Accepted')
    plt.scatter(negative['Test1'].values,negative['Test2'].values, c = 'r', marker = 'x', label = 'Rejected')
    plt.legend()
    plt.xlabel('Test1 Score')
    plt.ylabel('Test2 Score')
    
def predict(theta, x):
    probability = sigmoid(x.dot(theta.transpose()))
    return [1 if x >= 0.5 else 0 for x in probability]

    
def mapFeatures(data):
    x1 = data['Test1'].values
    x2 = data['Test2'].values
    degree = 5
    
    for i in range(1,degree):
        for j in range(i):
            data['F' + str(i) + str(j)] = np.power(x1,(i-j))*np.power(x2,j)
    return data

def mapfeature2(x1,x2):
    degree = 5
    x = np.ones(len(x1))
    x.shape = (len(x1),1)
    for i in range(1,degree):
        for j in range(i):
           x = np.append(x,np.power(x1,(i-j))*np.power(x2,j),axis = 1)
    return x

def plotboundary(theta):
    x1p = np.arange(-1,50,1.5)
    x1p.shape = (len(x1p),1)
    x2p = np.arange(-1,50,1.5)
    x2p.shape = (len(x2p),1)
    theta.shape = (theta.shape[0],1)
    z = np.zeros(shape = (len(x1p),len(x2p)))
    for i in range (len(x1p)):
        for j in range (len(x2p)):
            x = mapfeature2(x1p,x2p)
            temp = x.dot(theta)
            temp.shape = (len(x))
            z[[i,j]] = temp
    [X,Y] = np.meshgrid(x1p,x2p)
    plt.contour(X,Y,z,[0,1])
    plt.show()
    
def sigmoid(z):
    G = 1/(1 + np.exp(-z))
    G.shape = (len(z),1)
    return G

def costfunction(theta,x,y,learningRate):
    m = len(x)
    z = x.dot(theta.transpose())
    first = -y * np.log(sigmoid(z))
    second = np.multiply((1-y),np.log((1-sigmoid(z))))
    term = theta[1:(theta.shape[0]-1)]
    reg = (learningRate/(2*m))*np.sum((np.power(term,2)))
    cost = np.sum(first - second)/m + reg
    return cost

def gradientfunction(theta,x,y,learningRate):
    m = len(x)
    size = theta.shape[0]
    grad = np.zeros(size)
    grad.shape = (1,size)
   
    error = sigmoid(x.dot(theta.transpose())) - y
    
    for i in range(size):
        feature = np.array([x[:,i]]).transpose()
        term = np.multiply(error,feature)
        if i == 0:
            grad[0,i] = np.sum(term)/m
        else:
            grad[0,i] = np.sum(term)/m + (learningRate/m)*theta[i]
    return grad

data = pd.read_csv('ex2data2.txt',header=None,names=['Test1','Test2','Result'])

plotdata(data)

x = mapFeatures(data).loc[:,'F10':'F43'].values
y = data['Result'].values
y.shape = (len(y),1)
ones = np.ones([len(x),1])
x = np.append(ones,x,1)
learningRate = 1
theta = np.zeros(x.shape[1])
theta.shape = (1,x.shape[1])
#cost = costfunction(theta,x,y)
#grad = gradientfunction(theta,x,y)

result = opt.fmin_tnc(func=costfunction, x0=theta, fprime=gradientfunction, args=(x, y,learningRate))
cost = costfunction(result[0], x, y,learningRate)
theta = result[0]


#
#theta = np.array([ 0.35872309,  -3.22200653,  18.97106363,  -4.25297831,
#         18.23053189,  20.36386672,   8.94114455, -43.77439015,
#        -17.93440473, -50.75071857,  -2.84162964])
#predictions = predict(theta, x)
#correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
#accuracy = (sum(map(int, correct)) % len(correct))
#
#
#plotboundary(theta)




