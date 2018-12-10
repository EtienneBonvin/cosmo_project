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

def ridge_regression(y, X, lamb):
    aI = lamb * np.identity(X.shape[1])
    a = X.T.dot(X) + aI
    b = X.T.dot(y)
    return np.linalg.solve(a, b)

def build_poly(X, degree):
    poly = np.ones((len(X), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(X, deg)]
    return poly

def build_k_indices(y, k_fold, seed = 1):
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def split_train_test(y, x, k_indices, k):
    # get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    return y_te, y_tr, x_te, x_tr

def cross_validation(y, x, k_indices, k, degree, loss):
    y_te, y_tr, x_te, x_tr = split_train_test(y, x, k_indices, k)
    
    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)

    w = loss(y_tr, tx_tr)

    loss_tr = rmse(y_tr, tx_tr, w)
    loss_te = rmse(y_te, tx_te, w)
    return loss_tr, loss_te, w
