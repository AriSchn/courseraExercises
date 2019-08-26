# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 18:54:53 2019

@author: arschneider
"""
import numpy as np

def cost(theta):
    return 3*np.power(theta,3) + 2


theta = 1
e = 0.01
grad = (cost(theta + e) - cost(theta-e))/(2*e)