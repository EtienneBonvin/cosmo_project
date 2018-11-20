############################################
## Method implementations for regressions ##
############################################

import numpy as np
import math

#####################
## Error functions ##
#####################

def mse(y, X, w):
    e = y - X @ w
    return 0.5*np.mean(e**2)

def rmse(y, X, w):
    return math.sqrt(2*mse(y, X, w))

############################
## Good ol' least squares ##
############################

def least_squares(y, X):
    a = X.T.dot(X)
    b = X.T.dot(y)
    return np.linalg.solve(a, b)

##########################
## Polynomial expansion ##
##########################

def build_poly(X, degree):
    poly = np.ones((len(X), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(X, deg)]
    return poly
        
    
    
    
    
    
    