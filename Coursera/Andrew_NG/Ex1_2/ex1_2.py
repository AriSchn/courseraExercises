# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 16:58:16 2019

@author: arschneider
"""
def computeCost(X,Y,theta):
   m = len(X) 
   J = (1/(2*m))*np.dot(np.transpose((np.dot(X,theta)-Y)),(np.dot(X,theta)-Y))
   return J

def updateTheta(X,Y,theta):
    theta = inv(X.transpose().dot(X)).dot(X.transpose()).dot(Y)     
    return theta
    
    
    
    
import pandas as pd
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')


data = pd.read_csv('ex1data2.txt', header= None)
data.columns = ['size','bedrooms','price']
data = (data-data.mean())/data.std()
one_column = np.ones([len(data),1])

X = np.array([data['size'].values,data['bedrooms'].values])
X = np.transpose(X)
X = np.append(one_column,X,axis = 1)

Y = np.array(data['price'])
Y.shape = (len(data['price']),1)

theta = np.zeros(3)
theta.shape = (3,1)




theta = updateTheta(X,Y,theta)
cost = computeCost(X,Y,theta)



#plot the data 
fig= plt.figure()
ax = Axes3D(fig)
ax.scatter(data['size'].values,data['bedrooms'].values,data['price'].values)

ax.set_xlabel('Size')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price')


x_test = np.array([[1,data['size'].min(),data['bedrooms'].min()],[1,data['size'].max(),data['bedrooms'].max()]])   
z_plot = x_test.dot(theta)
z_plot.shape = (2)
x_plot = np.array([data['size'].min(),data['size'].max()])

y_plot = np.array([data['bedrooms'].min(),data['bedrooms'].max()])

ax.plot3D(x_plot,y_plot,z_plot,'r')

plt.show()