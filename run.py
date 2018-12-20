'''
Runs both machine and deep learning methods producing our best results.
Filename : run.py
Authors : Bonvin Etienne & Xavier Pantet
Creation date : 20/12/18
Last modified : 20/12/18
'''

import sys
import numpy as np
import math

import src.regressions as rg

import src.crowd as crowd
import src.supercrowd as sc
import src.collaborative_crowd as cc

########################
## Constants & Config ##
########################
DATA_FOLDER = "data/"
TRAIN_SET_PERC = 0.75
np.random.seed(1)

######################
## Machine Learning ##
######################
def ML():

    print("##########################################################")
    print("## Prediction using our best Machine Learning algorithm ##")
    print("##########################################################")
    print("Expected runtime: 1-2 mins...\n")

    print("Loading data...")
    X = np.load(DATA_FOLDER + "ML/x_train.npy")
    y = np.load(DATA_FOLDER + "ML/y.npy")
    print("Done!\n")

    print("Computing ridge regression with polynomial exp. of degree 1 and lambda = 1e-5")
    k_fold = 4
    k_indices = rg.build_k_indices(y, k_fold)
    lambda_ = 1e-5
    degree = 1
    ridge_lambda = lambda y, X: rg.ridge_regression(y, X, lambda_)

    losses_tr = []
    losses_te = []
    for k in range(k_fold):
        loss_tr, loss_te, _ = rg.cross_validation(y, X, k_indices, k, degree, ridge_lambda)
        losses_tr.append(loss_tr)
        losses_te.append(loss_te)

    print("Train error: {0:.2f}".format(np.mean(losses_tr)))
    print("Test error:  {0:.2f}".format(np.mean(losses_te)))
    
    

###################
## Deep Learning ##
###################
def DL():

    def rmse(pred, real, loop = True):
        '''
        Computes RMSE between predictions and real values
        :param : float[]
        :param : float[]
        :return : float
        '''
        if len(pred) != len(real):
            print("RMSE Error : Predictions and real values arrays do not have the same length, aborting.")
            return None
        
        if loop:
            mse = 0
            for i in range(len(pred)):
                mse += (pred[i] - real[i])**2
            return math.sqrt(mse/len(pred))
        else:
            # The creation of the array may produce memory error
            err = pred - real
            mse = err.T @ err
            return math.sqrt(2 * mse / len(pred))

    def prepare_DL_data():
        print("Prepare DL data...")
        X_red = np.load(DATA_FOLDER + "DL/feature_mat_radial_compression_normalized_red.npy")
        y = np.load(DATA_FOLDER + "DL/CSD500-r_train-H_total.npy")

        train_set_size = int(len(X_red) * TRAIN_SET_PERC)
        # Select random rows of the matrix for train / test set
        # Random seed for reproducibility 
        np.random.seed(100)
        train_idx = np.random.choice(len(X_red), size=train_set_size, replace = False)
        test_idx = [i for i in range(len(X_red)) if i not in train_idx]
        X_train_red = X_red[train_idx, :]
        X_test_red = X_red[test_idx, :]
        y_train = y[train_idx]
        y_test = y[test_idx]
        print("Train / Test split is {}/{}".format(TRAIN_SET_PERC * 100, (100 - TRAIN_SET_PERC*100)))
        print("X_train shape : {}".format(X_train_red.shape))
        print("y_train shape : {}".format(y_train.shape))
        print("X_test shape : {}".format(X_test_red.shape))
        print("y_test shape : {}\n".format(y_test.shape))
        return X_train_red, y_train, X_test_red, y_test
        
    print("#######################################################")
    print("## Prediction using our best Deep Learning algorithm ##")
    print("#######################################################")
    print("Expected runtime: 3-4 mins...\n")
    X_train, y_train, X_test, y_test = prepare_DL_data()
    
    supercrowd = sc.SuperCrowd()

    print("Load CollaborativeCrowd")
    cc1 = cc.CollaborativeCrowd(X_train, y_train, "CollabCrowd_2", nb_layers = 8, \
                        nb_neurons=156, regularization_factor=1e-6, validation_split = 0)
    cc1.restore()
    print()

    print("Load Simple Crowd")
    crowd_opt2 = crowd.Crowd(X_train, y_train, "Crowd_opt_2", nb_layers = 8, \
                        nb_neurons=156, regularization_factor=1e-6, validation_split = 0)
    crowd_opt2.restore()
    print()

    print("Combine in SuperCrowd and predict, please wait ...")
    supercrowd.append_crowd(cc1)
    supercrowd.append_crowd(crowd_opt2)
    
    print("Error of the supercrowd on the test set : {}".format(rmse(supercrowd.predict(X_test), y_test)))

##########
## Main ##
##########
if __name__ == "__main__":
    args = sys.argv[1:]
    
    if len(args) > 0:
        if args[0] == "ML":
            ML()
        elif args[0] == "DL":
            DL()
        else:
            print("Unknown argument {}, possible arguments are : \n\t- ML for Machine Learning \n\t- DL for Deep Learning".format(args[0]))
    else:
        DL()
