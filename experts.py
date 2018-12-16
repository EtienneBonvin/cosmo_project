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
from sklearn.cluster import KMeans


class Experts:
   
    def __init__(self, X, y, name, nb_layers=8, nb_neurons=128, activation='relu', regularization_factor=0.01, \
                optimizer_factor=0.001, loss='mse'):
        '''
        Creates a Crowd of Experts. The X matrix will be the training feature matrix, the y matrix are the predictions for the given X matrix. 
        Initially, the crowd is empty.
        The crowd should also be given a name in order to be saved and restored later.
        :param : two dimensional ndarray(float)
        :param : ndarray(float)
        :param : String
        '''
        self.experts = []
        self.gatingnet = None
        self.gatingnet_matrix = None
        self.gatingnet_trained = False
        self.X = X
        self.y = y
        self.crowd_name = name
        self.nb_layers = nb_layers
        self.nb_neurons = nb_neurons
        self.activation = activation
        self.regularization_factor = regularization_factor
        self.optimizer_factor = optimizer_factor
        self.loss = loss
        self.kmeans = None
        self.brain_assignments = None
        
        
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
    
    
    def __create_and_compile_gatingnet(self):
        '''
        Create the gating network used to specialize the brains. 
        :return : tf.keras.Sequential
        '''
        gatingnet = tf.keras.Sequential()
        for i in range(self.nb_layers):
            gatingnet.add(layers.Dense(self.nb_neurons, activation='relu', \
                                    kernel_regularizer=tf.keras.regularizers.l2(0.0001)))#0.0005 gives good results with 8 brains
        gatingnet.add(layers.Dense(len(self.experts), activation = 'softmax'))
        
        gatingnet.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
        return gatingnet
    
    
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
        
        self.gatingnet_trained = False
        
        
    def __build_gatingnet_matrix(self):
        '''
        Build the gating network matrix. Each sample is associated to the brain for which the prediction was the best.
        :return : ndarray(int)
        '''
        gatingnet_matrix = np.zeros(len(self.X))
        best_pred = np.empty(len(self.X))
        best_pred.fill(100)
        for i, expert in enumerate(self.experts):
            predictions = expert.predict(self.X)
            for j in range(len(predictions)):
                if abs(predictions[j] - self.y[j]) < best_pred[j]:
                    best_pred[j] = abs(predictions[j] - self.y[j])
                    gatingnet_matrix[j] = i
        self.gatingnet_matrix = gatingnet_matrix
        
    
    def __get_cluster_i(self, assignments, i):
        '''
        Return all the samples in the cluster number i as well as the corresponding values.
        The list (index, cluster_index) should be given as argument.
        :param : List(int, int)
        :return : 2 dimensional ndarray(float), ndarray(float)
        '''
        cluster = []
        vals = []
        for j, a in enumerate(assignments):
            if a == i:
                cluster.append(self.X[j])
                vals.append(self.y[j])
        return np.asarray(cluster), np.asarray(vals)
    
    
    def __replace_max(self, min_loss, best_brain_idx, loss, j):
        '''
        Replace the value of the maximum error in the array as well as the corresponding index of the given loss of the (loss, index) is less than the ones in the array.
        Changes are made in-place.
        :param : List(float)
        :param : List(int)
        :param : float
        :param : int
        '''
        max_found = 0
        max_idx = -1
        for i, l in enumerate(min_loss):
            if l > max_found:
                max_found = l
                max_idx = i
                
        if loss < max_found:
            min_loss[max_idx] = loss
            best_brain_idx[max_idx] = j
    
    
    def clusterize(self, loss_func, nb_clusters):
        '''
        Create the clusters for the data and select the optimal brains for the predictions.
        :param : function (List(float), List(float)) -> float
        :param : int
        '''
        print("##############")
        print("# Clustering #")
        print("##############")
        if self.kmeans == None:
            self.kmeans = KMeans(n_clusters=nb_clusters, random_state=0, verbose=1).fit(self.X)
        assignments = self.kmeans.labels_
        self.brain_assignments = {}
        for i in range(nb_clusters):
            cluster_i, vals_i = self.__get_cluster_i(assignments, i)
            best_brain_idx = np.zeros(self.size() // nb_clusters)
            best_brain_idx.fill(-1)
            min_loss = np.zeros(self.size() // nb_clusters)
            min_loss.fill(1000)
            for j in range(self.size()):
                loss = loss_func(vals_i, self.experts[j].predict(cluster_i))
                self.__replace_max(min_loss, best_brain_idx, loss, j)
            print("Cluster {} ({} entities) : brains : {}, losses : {}".format(i, len(cluster_i), best_brain_idx, min_loss))
            self.brain_assignments[i] = best_brain_idx
        
        
    def train_gatingnet(self):
        '''
        Create and train the gating network. The accuracy of the network depends on the categorizable aspect of the data.
        '''
        print("######################")
        print("# Creating gatingnet #")
        print("######################")
        self.gatingnet = self.__create_and_compile_gatingnet()
        
        if not os.path.exists("session/"+self.name()):
            os.makedirs("session/"+self.name())
        
        checkpoint_path = "session/"+self.name()+"/gatingnet"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        
        # Create checkpoint callback
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only = True, verbose = 1)
        
        EPOCHS = 200
        BATCH_SIZE = 32
        VALIDATION_SPLIT = 0.1
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=20)
        
        if self.gatingnet_matrix is None:
            self.__build_gatingnet_matrix()
        
        
        self.gatingnet.fit(self.X, self.gatingnet_matrix, \
                  epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split = VALIDATION_SPLIT, \
                  callbacks=[early_stop, cp_callback])
        
        self.gatingnet_trained = True
        
        
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
        
        if not self.gatingnet_trained:
            self.train_gatingnet()
        
        predictions = []
        for expert in self.experts:
            predictions.append(expert.predict(X_test, batch_size=32))
            
        weights = self.gatingnet.predict(X_test, batch_size=32)
        
        # Weighted average prediction
        final_prediction = []
        for i in range(len(X_test)):
            pred = 0
            for j in range(len(self.experts)):
                pred += predictions[j][i][0] * weights[i][j]
            final_prediction.append(pred)
        return np.asarray(final_prediction)
    
    
    def predict_kmeans(self, X_test, loss_func, nb_clusters):
        '''
        Predict the values using the k-means clustering to determine best brains for prediction.
        :param : 2 dimensional ndarray(float)
        :param : func (List(float), List(float)) -> float
        :param : int
        :return : ndarray(float)
        '''
        if self.brain_assignments == None:
            self.clusterize(loss_func, nb_clusters)
            
        clusters_assignment = self.kmeans.predict(X_test)
        brain_assignments = map(lambda c : self.brain_assignments[c], clusters_assignment)
        
        brain_predictions = np.asarray([expert.predict(X_test) for expert in self.experts])
            
            
        final_predictions = []
        for i, j in enumerate(brain_assignments):
            pred = []
            for idx in j:
                pred.append(brain_predictions[int(idx)][i][0])
            final_predictions.append(np.mean(pred))
        return final_predictions
        
    
    
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
    
    
    def get_gatingnet(self, X_test):
        '''
        Return the predictions of the gating network on the given sample matrix if the gating network has already been created and trained.
        The result if a vector where the value at index i is the probability that the brain i should be choosen for the prediction.
        :param : 2 dimensional ndarray(float)
        :return : ndarray(float)
        '''
        if self.gatingnet_trained:
            return self.gatingnet.predict(X_test, batch_size=32)
        else:
            print("Please train the gatingnet before displaying it.")
            
            
    def get_gatingnet_matrix(self):
        '''
        Return the gating network matrix if it has already been created.
        The result is a vector where the value at index i is the number of the brain which performed best on sample i.
        :return : ndarray(int)    
        '''
        if self.gatingnet_trained:
            return self.gatingnet_matrix
        else:
            print("Please train the gatingnet to access the gatingnet matrix.")
    
    
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