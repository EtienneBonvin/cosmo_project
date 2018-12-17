'''
Crowd class. Allow to instantiate a crowd of neural networks, significantly decreasing the error of a single, highly trained neural network.
Each neural networks is trained independantly. The prediction are then made by averaging all the predictions of the neural networks of the crowd.

Filename : crowd.py
Author : Bonvin Etienne
Creation date : 03/12/18
Last modified : 17/12/18
'''


import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import timeit


class Crowd:
   
    def __init__(self, X, y, name, nb_layers=8, nb_neurons=128, activation='relu', regularization_factor=0.01, \
                optimizer_factor=0.001, loss='mse', validation_split=0.1):
        '''
        Creates a Crowd. The X matrix will be the training feature matrix, the y matrix are the predictions for the given X matrix. 
        Initially, the crowd is empty.
        The crowd should also be given a name in order to be saved and restored later.
        :param : two dimensional ndarray(float)
        :param : ndarray(float)
        :param : String
        '''
        self.models = []
        self.X = X
        self.y = y
        self.crowd_name = name
        self.nb_layers = nb_layers
        self.nb_neurons = nb_neurons
        self.activation = activation
        self.regularization_factor = regularization_factor
        self.optimizer_factor = optimizer_factor
        self.loss = loss
        self.validation_split = validation_split
        self.stacker = None
        
        
    def size(self):
        '''
        Give the size of the crowd. I.e the number of neural networks in it.
        :return : int
        '''
        return len(self.models)
    
    
    def __create_and_compile_model(self):
        '''
        Create a default neural network model and compile it.
        :return : tensorflow model
        '''
        model = tf.keras.Sequential()
        for i in range(self.nb_layers):
            model.add(layers.Dense(self.nb_neurons, activation=self.activation, \
                                   kernel_regularizer=tf.keras.regularizers.l2(self.regularization_factor)))
        # Last layer represent the electromagnetic shielding, our prediction
        model.add(layers.Dense(1, activation='relu'))

        model.compile(optimizer=tf.train.AdamOptimizer(self.optimizer_factor),
                  loss=self.loss,
                  metrics=['mae'])
        return model
    
    
    def __train_one_entity(self):
        '''
        Creates a new entity (a new neural network) and train it on the train data. 
        The created neural networks is then added to the crowd.
        '''
        model = self.__create_and_compile_model()
        
        if not os.path.exists("session/"+self.name()):
            os.makedirs("session/"+self.name())
        
        checkpoint_path = "session/"+self.name()+"/"+str(self.size())
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create checkpoint callback
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only = True, verbose = 1)
        
        EPOCHS = 200
        BATCH_SIZE = 32
        VALIDATION_SPLIT = self.validation_split
        
        # Create ealy stop callback to stop earlier loss has converged to avoid overfitting.
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss' if VALIDATION_SPLIT > 0 else 'loss', \
                                                      patience=20)
        
        model.fit(self.X, self.y, \
                  epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split = VALIDATION_SPLIT, \
                  callbacks=[early_stop, cp_callback])
        self.models.append(model)
        
        
    def name(self):
        '''
        Generates a unique name defining the crowd.
        '''
        return "{}_{}_{}_{}_{}_{}_{}".format(self.crowd_name, self.nb_layers, self.nb_neurons, self.activation, self.regularization_factor, self.optimizer_factor, self.loss)
    
    
    def get_models(self):
        '''
        Give the list of all the models of the crowd.
        :return : list(tensorflow.keras.models)
        '''
        return self.models
        
        
    def train_new_entities(self, number = 1):
        '''
        Add a given number of entity to the crowd. By default, add only one entity. 
        :param : int
        '''
        for i in range(number):
            self.__train_one_entity()
        
        
    def predict(self, X_test):
        '''
        Predict the output of the given matrix.
        :param : two dimensional ndarray(float)
        :return : ndarray(float)
        '''
        return np.mean([model.predict(X_test, batch_size=32) for model in self.models], axis = 0)
    
    
    def predict_stacked(self, X_test):
        stacker = self.__create_and_compile_model()
        
        EPOCHS = 200
        BATCH_SIZE = 32
        VALIDATION_SPLIT = self.validation_split
        
        # Create ealy stop callback to stop earlier loss has converged to avoid overfitting.
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss' if VALIDATION_SPLIT > 0 else 'loss', \
                                                      patience=20)
        
        predictions = np.asarray([model.predict(self.X) for model in self.models])
        print(self.X[0].shape, predictions.T[0].shape)
        new_X = np.asarray([np.concatenate([self.X[i], predictions.T[0][i]]) for i in range(len(self.X))])
        stacker.fit(new_X, self.y, \
                  epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split = VALIDATION_SPLIT, \
                  callbacks=[early_stop])
        
        test_pred = np.asarray([model.predict(X_test) for model in self.models])
        new_X_test = np.asarray([np.concatenate([X_test[i], test_pred.T[0][i]]) for i in range(len(X_test))])
        return stacker.predict(new_X_test, batch_size = 32)
        
    
    
    def subcrowd_predict(self, X_test, number):
        '''
        Predict the output of the given matrix using only a given number of entities.
        :param : two dimensional ndarray(float)
        :param : int
        :return : ndarray(float)
        '''
        return np.mean([model.predict(X_test, batch_size=32) for model in self.models[:number]], axis = 0)
    
    
    def plot_crowd_error(self, X_test, y_test, func):
        '''
        Compute error in prediction the subcrowds and plot the error made as a function of the entities used.
        The error is computed using the given function.
        :param : two dimensional ndarray(float)
        :param : ndarray(float)
        :param : function
        '''
        error = []
        for i in range(len(self.models) - 1):
            pred = self.subcrowd_predict(X_test, i + 1)
            err = func(pred, y_test)
            error.append(err)
        plt.plot(list(range(1, self.size())), error)
        plt.title("Error vs. number of entities used for prediction")
        plt.xlabel("Number of entities used")
        plt.ylabel("Error")
        plt.show()
        
        
    def plot_crowd_pred_time(self, X_test):
        '''
        Plot the prediction time with respect to the number of entites used to make the prediction.
        :param : 2 dimensional ndarray(float)
        '''
        times = []
        for i in range(len(self.models) - 1):
            start = timeit.default_timer()
            self.subcrowd_predict(X_test, i + 1)
            stop = timeit.default_timer()
            times.append(stop - start)
        plt.plot(list(range(1, self.size())), times)
        plt.title("Time vs. number of entities used for prediction")
        plt.xlabel("Number of entities used")
        plt.ylabel("Time (in seconds)")
        plt.show()
        
        
    def plot_error_dist_on_sample(self, sample, y_sample, func, show = True, bins = 4):
        '''
        Plot error distribution on a single sample over all models as an histogram.
        The error is computed using the given function.
        :param : ndarray(float)
        :param : float
        :param : function
        :param : boolean
        '''
        error = []
        for m in self.models:
            pred = m.predict(np.asarray([sample]))
            error.append(func(np.asarray([pred]), np.asarray([y_sample])))
        plt.hist(error, bins = bins)
        plt.title("Distribution of error over models for single sample")
        plt.xlabel("Error")
        plt.ylabel("Number of models")
        if show:
            plt.show()
    
    
    def restore(self):
        '''
        Restore a crowd from files. The path of the directory corresponding to the crowd should be result/name.
        If this directory doesn't exist, the crowd will not be restored.
        '''
        if os.path.exists("session/"+self.name()):
            counter = 0
            while os.path.exists("session/"+self.name()+"/"+str(counter)+".index"):
                
                model = self.__create_and_compile_model()
                model.load_weights("session/"+self.name()+"/"+str(counter))
                self.models.append(model)
                counter += 1
                
            print("Recovered {} entities from {}".format(counter, "session/"+self.name()+"/"+str(counter)))
            
            
        else:
            print("No directory with name {}".format("session/"+self.name()))