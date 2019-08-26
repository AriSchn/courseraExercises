# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:23:19 2019

@author: arschneider
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 18:54:53 2019

@author: arschneider
"""

from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def shownumber(index):
    
    number = x[index,:].reshape(20,20).swapaxes(1,0)
    plt.figure(1)
    plt.imshow(number,cmap='gray', vmin=0, vmax=1)
    
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def h(thetas,x):
    
    z1 = np.dot(x,thetas['theta1'].transpose())
    a1 = sigmoid(z1)
    x2 = np.append(np.ones(shape = (a1.shape[0],1)),a1,1)
    z2 = np.dot(x2,thetas['theta2'].transpose())
    a2 = sigmoid(z2)
    return a2

def cost(y,h,thetas,lamb):
#    k = h.shape[1]
    m = h.shape[0]
    term1 = np.multiply(np.multiply(y,-1),np.log(h))
    term2 = np.multiply(((np.ones(shape=(y.shape[0],1))) - y),np.log(np.ones(shape = (h.shape[0],h.shape[1]))-h))
    J =  np.sum(term1 - term2)/m
    term3 =  lamb*(np.sum(np.power(thetas['theta1'][:,1:thetas['theta1'].shape[1]],2)) + np.sum(np.power(thetas['theta2'][:,1:thetas['theta2'].shape[1]],2)))/(2*m)
    
    J = J + term3
    return J


data = loadmat('ex4data1.mat')
weights = loadmat('ex4weights.mat')

thetas = {"theta1": weights['Theta1'],
          "theta2": weights['Theta2']
        
        }

#Get Examples
x = data['X']
y = data ['y']
lamb = 1;
#add the column of ones to x
x = np.append(np.ones(shape = (x.shape[0],1)),x,1)

#Transform each yi in a vector 
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)
    
#compute forward propagation    
h = h(thetas,x)
#comput cost
J = cost(y,h,thetas,lamb)

