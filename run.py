import sys
import numpy as np
import math

import crowd
import supercrowd as sc
import collaborative_crowd as cc


#############
# Constants #
#############


DATA_FOLDER = "data/"
TRAIN_SET_PERC = 0.75


###########
# Helpers #
###########


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


####################
# Machine Learning #
####################


def ML():
    print("Let's do some ML !")
    

#################
# Deep Learning #
#################

def prepare_DL_data():
    print("Prepare DL data...")
    X_red = np.load(DATA_FOLDER + "feature_mat_radial_compression_normalized_red.npy")
    y = np.load(DATA_FOLDER + "CSD500-r_train-H_total.npy")

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
    
    
def DL():
    print("#####################################################")
    print("# Prediction using our best Deep Learning algorithm #")
    print("#####################################################\n")
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