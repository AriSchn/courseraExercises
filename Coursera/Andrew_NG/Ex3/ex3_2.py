# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 22:21:42 2019

@author: arschneider
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

data = loadmat('ex3data1.mat')
data_thetas = loadmat('ex3weights.mat')

theta1 = data_thetas['Theta1']
theta2 = data_thetas['Theta2']

x = data['X']
y = data['y']

x = np.append(np.ones(shape=(5000,1)),x,1)

z2 = np.dot(x,theta1.transpose())                                                                                                    

a2= sigmoid(z2)


a2_b = np.append(np.ones(shape = (5000,1)),a2,1)

z3 = np.dot(a2_b,theta2.transpose())

a3 = sigmoid(z3)


h_argmax = np.argmax(a3, axis=1)

h_argmax = h_argmax + 1

h_argmax.shape = (5000,1)

correct = (y==h_argmax)
correct = correct.astype(int)
accuracy = (sum(map(int, correct)) / float(len(correct)))
print(accuracy*100)
