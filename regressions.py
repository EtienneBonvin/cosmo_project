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
    return 0.5*np.mean(e ** 2)

def rmse(y, X, w):
    return math.sqrt(2*mse(y, X, w))

############################
## Good ol' least squares ##
############################

def least_squares(y, X):
    a = X.T.dot(X)
    b = X.T.dot(y)
    return np.linalg.solve(a, b)

######################
## Ridge regression ##
######################

def ridge_regression(y, tx, lamb):
    aI = lamb * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)

################################
## Loss functions & gradients ##
################################

def sign(x):
    if x == 0:
        return 0
    else:
        return 1 if x > 0 else -1
    
def mse_loss(y, X, w):
    return mse(y, X, w)

def mse_stoch_grad(y, X, w):
    e = y - X.T @ w
    return (-X).T * e
        
def mae(y, X, w, lambda_ = 0):
    e = y - X @ w
    return np.mean(np.abs(e)) + lambda_ * np.sum(np.square(w))

def mae_stoch_grad(y, X, w, lambda_ = 0):
    e = y - X.T @ w
    s = np.vectorize(sign)
    return -1/len(X) * X.T * s(e) + 2*lambda_*w

def lasso(y, X, w, lambda_):
    return mse(y, X, w) + lambda_ * np.sum(np.absolute(w))

def lasso_stoch_grad(y, X, w, lambda_):
    e = y - X.T @ w
    s = np.vectorize(sign)
    return (-X.T * e) + lambda_ * s(w)

#################################
## Stochastic Gradient Descent ##
#################################

def sample_from(y, X, n = 1):
    assert(n <= len(y))
    y_shuf = y
    X_shuf = X
    np.random.shuffle(y_shuf)
    np.random.shuffle(X_shuf)
    return y_shuf[:n], X_shuf[:n]
    
def stochastic_gradient_descent(y, X, initial_w, max_iters, gamma, loss_function, grad_loss_function, batch_size = 1, detail = False):
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        y_batch, X_batch = sample_from(y, X, n = batch_size)
        grad = np.mean([grad_loss_function(y_i, X_i, w) for (y_i, X_i) in zip(y_batch, X_batch)], axis = 0)
        w = w - (gamma/(n_iter+1)) * grad
        loss = loss_function(y, X, w)
        ws.append(w)
        losses.append(loss)
        
    if detail:
        return ws, losses
    else:
        return ws[np.argmin(losses)]

##########################
## Polynomial expansion ##
##########################

def build_poly(X, degree):
    poly = np.ones((len(X), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(X, deg)]
    return poly

############
## k-fold ##
############

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