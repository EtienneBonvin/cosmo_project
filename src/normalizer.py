'''
Script used to normalize a matrix representing some data.

Filename : normalizer.py
Author : Etienne Bonvin
Creation date : 20/11/18
Last modified : 20/11/18
'''
import numpy as np

DATA_FOLDER = 'data/'

def load_matrix():
    X = np.load(DATA_FOLDER + "feature_mat_radial_compression.npy")
    return X

def normalize(matrix, chunks_size = 128):
    print(matrix.shape)
    nb_chunks = len(matrix[0]) // chunks_size 
    print(nb_chunks, chunks_size * nb_chunks)
    # Manage first chunks
    for i in range(nb_chunks):
        sample_data = matrix.T[i * chunks_size : (i+1) * chunks_size]
        mean_sample = sample_data.mean(axis = 1)
        std_sample = sample_data.std(axis = 1)
        for j, std in enumerate(std_sample):
            if std == 0:
                print("Variance 0 found at index {}".format(i * chunks_size + j))
                # Replace by default value
                std_sample[j] = 1
        
        if len(mean_sample) != chunks_size or len(std_sample) != chunks_size:
            print("An error occured, length of mean and std doesn't match")
            
        normalized_sample = (sample_data.T - mean_sample) / std_sample
        
        for j in range(len(normalized_sample.T)):
            matrix[:, i * chunks_size + j] = normalized_sample.T[j]
            
        print("{}% done".format((i + 1)*chunks_size / len(matrix[0]) * 100))
    # Manage last chunk
    sample_data = matrix.T[nb_chunks * chunks_size : ]
    mean_sample = sample_data.mean(axis = 1)
    std_sample = sample_data.std(axis = 1)
    
    for j, std in enumerate(std_sample):
            if std == 0:
                print("Variance 0 found at index {}".format(nb_chunks * chunks_size + j))
                # Replace by default value
                std_sample[j] = 1
    
    normalized_sample = (sample_data.T - mean_sample) / std_sample

    for j in range(len(normalized_sample.T)):
        matrix[:, nb_chunks * chunks_size + j] = normalized_sample.T[j]

    print("100% done")
    return matrix
    
    

def save_matrix(matrix, filename):
    np.save(filename, matrix)

if __name__ == "__main__":
    print("Starting ...")
    X = load_matrix()
    print("Matrix loaded")
    X_norm = normalize(X)
    print("Matrix normalized")
    save_matrix(X_norm, DATA_FOLDER + "feature_mat_radial_compression_normalized.npy")
    print("Matrix saved")
    print("Job finished")
    
    
    