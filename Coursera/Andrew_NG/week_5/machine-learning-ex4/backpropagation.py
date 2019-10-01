# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:03:50 2019

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

def gradSigm(z):
    return np.multiply(sigmoid(z),(1-sigmoid(z)))

def h(thetas,x):
    
      
    z2 = np.dot(x,thetas['theta1'].transpose())
    a2 = sigmoid(z2)
    a2 = np.append(np.ones(shape = (a2.shape[0],1)),a2,1)
    z3 = np.dot(a2,thetas['theta2'].transpose())
    a3 = sigmoid(z3)
    
    
    return z2,a2,z3,a3

def cost(y,h,thetas,lamb):
#    k = h.shape[1]
    m = h.shape[0]
    term1 = np.multiply(np.multiply(y,-1),np.log(h))
    term2 = np.multiply(((np.ones(shape=(y.shape[0],1))) - y),np.log(np.ones(shape = (h.shape[0],h.shape[1]))-h))
    J =  np.sum(term1 - term2)/m
    term3 =  lamb*(np.sum(np.power(thetas['theta1'][:,1:thetas['theta1'].shape[1]],2)) + np.sum(np.power(thetas['theta2'][:,1:thetas['theta2'].shape[1]],2)))/(2*m)
    
    J = J + term3
    return J

def randomInit(lin,lout,eps):
    theta_init = np.random.random((lin,1+lout))
    theta_init = theta_init*2*eps - eps
    return theta_init

def backpropagation(x,z2,a2,z3,h,y,m,thetas,lamb):
    
    
    
    delta1 = np.zeros(shape = (thetas['theta1'].shape[0],thetas['theta1'].shape[1]));
    delta2 = np.zeros(shape = (thetas['theta2'].shape[0],thetas['theta2'].shape[1]));
    [z2,a2,z3,h] = h(thetas,x)
    J = cost(y,h,thetas,lamb)
    for t in range(m):
        
        
        a1t = x[t,:] # 1x401
        a1t.shape = [1,401]
        z2t = z2[t,:] # 1 x 25
        z2t.shape = [1,25]
        a2t = a2[t,:] # 1 x 26
        a2t.shape = [1,26]
        z3t = z3[t,:] # 1 x 10
        z3t.shape = [1,10]
        ht = h[t,:] # 1 x 10
        ht.shape = [1,10]
        yt = y[t,:] # 1 x 10
        yt.shape = [1,10]
        
        
        d3t = ht - yt
        
        z2t = np.append(np.ones(shape = (1,1)),z2t,1)
        d2t = np.transpose(np.dot(np.transpose(thetas['theta2']),np.transpose(d3t)))
        
        delta1 = delta1 + np.dot(np.transpose(d2t[:,1:]),a1t)
        delta2 = delta2 + np.dot(np.transpose(d3t),a2t)
        
        delta1 = delta1/m
        delta2 = delta2/m
        
        #adding the regularization term
        
        delta1[:,1:] = delta1[:,1:] + (thetas['theta1'][:,1:]*lamb)/m
        delta2[:,1:] = delta2[:,1:] + (thetas['theta2'][:,1:]*lamb)/m
        
        
    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
        
    return J, grad

data = loadmat('ex4data1.mat')
weights = loadmat('ex4weights.mat')

thetas = {"theta1": weights['Theta1'],
          "theta2": weights['Theta2']
        
        }


#theta1 = randomInit(25,400,0.12)
#theta2 = randomInit(10,25,0.12)
#
#thetas = {"theta1": theta1,
#          "theta2": theta2
#          }
#Get Examples
x = data['X']
y = data ['y']

#add the column of ones to x
x = np.append(np.ones(shape = (x.shape[0],1)),x,1)

#Transform each yi in a vector 
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)
    
##compute forward propagation    
#[z2,a2,z3,h] = h(thetas,x)
##comput cost
#J = cost(y,h)

m = y.shape[0]

#define learning rate
lamb = 1

for i in range (10):
 [J,grad] = backpropagation(x,z2,a2,z3,h,y,m,thetas,lamb)
