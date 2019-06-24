# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 20:04:36 2019

@author: arschneider
"""

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#plot the data 
fig= plt.figure()
ax = Axes3D(fig)


x = np.array([-1,3])
y = np.array([-1,3])
z = np.array([0,2.5])


x1 = np.arange(1,10)

ax.plot3D(x,y,z)

plt.show()