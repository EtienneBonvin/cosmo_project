'''
Experts class. Allow to instantiate a crowd of experts neural networks, significantly decreasing the error of a single, highly trained neural network.
Each neural networks is trained independantly and is supposed to be an expert for a subset of the samples. The prediction is then made using the best experts.

Filename : experts.py
Creation date : 06/12/18
Last modified : 06/12/18
'''


import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import matplotlib.pyplot as plt


class Experts:
   
    def __init__(self, X, y, name, nb_layers=8, nb_neurons=128, activation='relu', regularization_factor=0.01, \
                optimizer_factor=0.001, loss='mse'):
        '''
        Creates a Crowd. The X matrix will be the training feature matrix, the y matrix are the predictions for the given X matrix. 
        Initially, the crowd is empty.
        The crowd should also be given a name in order to be saved and restored later.
        :param : two dimensional ndarray(float)
        :param : ndarray(float)
        :param : String
        '''
        self.experts = []
        self.gateway = None
        self.gateway_matrix = None
        self.gateway_trained = False
        self.X = X
        self.y = y
        self.crowd_name = name
        self.nb_layers = nb_layers
        self.nb_neurons = nb_neurons
        self.activation = activation
        self.regularization_factor = regularization_factor
        self.optimizer_factor = optimizer_factor
        self.loss = loss
        
        
    def size(self):
        '''
        Give the size of the crowd. I.e the number of neural networks in it.
        :return : int
        '''
        return len(self.experts)
    
    
    def __create_and_compile_expert(self):
        '''
        Create a default neural network expert and compile it.
        :return : tensorflow expert
        '''
        expert = tf.keras.Sequential()
        for i in range(self.nb_layers):
            expert.add(layers.Dense(self.nb_neurons, activation=self.activation, \
                                   kernel_regularizer=tf.keras.regularizers.l2(self.regularization_factor)))
        # Last layer represent the electromagnetic shielding, our prediction
        expert.add(layers.Dense(1, activation='relu'))

        expert.compile(optimizer=tf.train.AdamOptimizer(self.optimizer_factor),
                  loss=self.loss,
                  metrics=['mae'])
        return expert
    
    
    def __create_and_compile_gateway(self):
        gateway = tf.keras.Sequential()
        for i in range(self.nb_layers):
            gateway.add(layers.Dense(self.nb_neurons, activation='relu', \
                                    kernel_regularizer=tf.keras.regularizers.l2(0.0005)))#0.0005 gives good results with 8 brains
        gateway.add(layers.Dense(len(self.experts), activation = 'softmax'))
        
        gateway.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
        return gateway
    
    
    def __train_one_entity(self):
        '''
        Creates a new entity (a new neural network) and train it on the train data. 
        The created neural networks is then added to the crowd.
        '''
        expert = self.__create_and_compile_expert()
        
        if not os.path.exists("session/"+self.name()):
            os.makedirs("session/"+self.name())
        
        checkpoint_path = "session/"+self.name()+"/"+str(self.size())
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create checkpoint callback
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only = True, verbose = 1)
        
        EPOCHS = 200
        BATCH_SIZE = 32
        VALIDATION_SPLIT = 0.1
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
        
        expert.fit(self.X, self.y, \
                  epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split = VALIDATION_SPLIT, \
                  callbacks=[early_stop, cp_callback])
        self.experts.append(expert)
        
        self.gateway_trained = False
        
        
    def __build_gateway_matrix(self):
        gateway_matrix = np.zeros(len(self.X))
        best_pred = np.empty(len(self.X))
        best_pred.fill(100)
        for i, expert in enumerate(self.experts):
            predictions = expert.predict(self.X)
            for j in range(len(predictions)):
                if abs(predictions[j] - self.y[j]) < best_pred[j]:
                    best_pred[j] = abs(predictions[j] - self.y[j])
                    gateway_matrix[j] = i
        self.gateway_matrix = gateway_matrix
        
        
    def train_gateway(self):
        
        print("####################")
        print("# Creating gateway #")
        print("####################")
        self.gateway = self.__create_and_compile_gateway()
        
        if not os.path.exists("session/"+self.name()):
            os.makedirs("session/"+self.name())
        
        checkpoint_path = "session/"+self.name()+"/gateway"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        
        # Create checkpoint callback
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only = True, verbose = 1)
        
        EPOCHS = 200
        BATCH_SIZE = 32
        VALIDATION_SPLIT = 0.1
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=20)
        
        if self.gateway_matrix is None:
            self.__build_gateway_matrix()
        
        
        self.gateway.fit(self.X, self.gateway_matrix, \
                  epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split = VALIDATION_SPLIT, \
                  callbacks=[early_stop, cp_callback])
        
        self.gateway_trained = True
        
        
    def name(self):
        '''
        Generates a name defining the crowd.
        '''
        return "{}_{}_{}_{}_{}_{}_{}".format(self.crowd_name, self.nb_layers, self.nb_neurons, self.activation, self.regularization_factor, self.optimizer_factor, self.loss)
        
        
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
        
        if not self.gateway_trained:
            self.train_gateway()
        
        predictions = []
        for expert in self.experts:
            predictions.append(expert.predict(X_test, batch_size=32))
            
        weights = self.gateway.predict(X_test, batch_size=32)
        
        final_prediction = []
        for i in range(len(X_test)):
            pred = 0
            for j in range(len(self.experts)):
                pred += predictions[j][i][0] * weights[i][j]
            final_prediction.append(pred)
        return np.asarray(final_prediction)
    
    
    def subcrowd_predict(self, X_test, number):
        '''
        Predict the output of the given matrix using only a given number of entities.
        :param : two dimensional ndarray(float)
        :param : int
        :return : ndarray(float)
        '''
        predictions = []
        for expert in self.experts[:number]:
            predictions.append(expert.predict(X_test, batch_size=32))
        return np.mean(predictions, axis=0)
    
    
    def get_gateway(self, X_test):
        if self.gateway_trained:
            return self.gateway.predict(X_test, batch_size=32)
        else:
            print("Please train the gateway before displaying it.")
            
            
    def get_gateway_matrix(self):
        if self.gateway_trained:
            return self.gateway_matrix
        else:
            print("Please train the gateway to access the gateway matrix.")
    
    
    def plot_crowd_error(self, X_test, y_test, func):
        '''
        Compute error in prediction the subcrowds and plot the error made as a function of the entities used.
        The error is computed using the given function.
        :param : two dimensional ndarray(float)
        :param : ndarray(float)
        :param : function
        '''
        error = []
        for i in range(len(self.experts) - 1):
            pred = self.subcrowd_predict(X_test, i + 1)
            err = func(pred, y_test)
            error.append(err)
        plt.plot(error)
        plt.title("Error vs. number of entities used for prediction")
        plt.xlabel("Number of entities used")
        plt.ylabel("Error")
        plt.show()
        
        
    def plot_error_dist_on_sample(self, sample, y_sample, func, show = True, bins = 4):
        '''
        Plot error distribution on a single sample over all experts as an histogram.
        The error is computed using the given function.
        :param : ndarray(float)
        :param : float
        :param : function
        :param : boolean
        '''
        error = []
        for m in self.experts:
            pred = m.predict(np.asarray([sample]))
            error.append(func(np.asarray([pred]), np.asarray([y_sample])))
        plt.hist(error, bins = bins)
        plt.title("Distribution of error over experts for single sample")
        plt.xlabel("Error")
        plt.ylabel("Number of experts")
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
                
                expert = self.__create_and_compile_expert()
                expert.load_weights("session/"+self.name()+"/"+str(counter))
                self.experts.append(expert)
                counter += 1
                
            print("Recovered {} entities from {}".format(counter, "session/"+self.name()+"/"+str(counter)))
            
            
        else:
            print("No directory with name {}".format("session/"+self.name()))