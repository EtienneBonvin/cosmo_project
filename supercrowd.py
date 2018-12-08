import numpy as np


class SuperCrowd:
    '''
    Combines crowd to reduce bias in estimation.
    '''
    def __init__(self):
        self.crowds = []
        
    def append_crowd(self, crowd):
        '''
        Add a crowd to the supercrowd.
        :param : Crowd
        '''
        self.crowds.append(crowd)
        
        
    def size(self):
        '''
        Return the total size of the SuperCrowd, i.e the sum of the sizes of all the crowds in the SuperCrowd.
        return : int
        '''
        total_size = 0
        for crowd in self.crowds:
            total_size += crowd.size()
        return total_size
        
        
    def predict(self, X):
        '''
        Predict the values of the given input matrix. The prediction created is the average of all the predictions of the crowds in the SuperCrowd.
        :param : two dimensional ndarray(float)
        :return : ndarray(float)
        '''
        predictions = []
        for crowd in self.crowds:
            predictions.append(crowd.predict(X))
        return np.mean(predictions, axis = 0)