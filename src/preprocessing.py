import numpy as np
import pandas as pd
import random
from regressions import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

DATA_FOLDER = "data/"
X = np.load(DATA_FOLDER + "feature_mat_radial_compression.npy")
y = np.load(DATA_FOLDER + "CSD500-r_train-H_total.npy")

x_df = pd.DataFrame(X)

def normalize(df):
    df_tmp = df.copy()
    df_tmp =(df_tmp-df_tmp.mean())/df_tmp.std()
    df_tmp = df_tmp.drop(15960, axis=1)
    return df_tmp

def add_jitter(df, y, perc_sample, perc_col):
    means = [df[j].mean() for j in df]
    df_tmp = df.copy()
    y_tmp = y.copy()
    ids = random.sample(range(df_tmp.shape[0]), int(df_tmp.shape[0]*perc_sample))
    for id_ in ids:
        new_sample = df_tmp.iloc[[id_]].copy()
        col = random.sample(range(df_tmp.shape[1]), int(df_tmp.shape[1]*perc_col))
        for j in col:
            new_sample[j] = new_sample[j] + 0.01*means[j]
        df_tmp = df_tmp.append(new_sample, ignore_index=True)
        y_tmp = np.append(y_tmp, y_tmp[id_])
    return df_tmp, y_tmp        

def PCA_ML_object(df):
    i_star = 4500
    pca_ML = PCA(n_components=i_star)
    principalComponents_ML = pca_ML.fit_transform(df, df.shape)
    return principalComponents_ML
    
def PCA_DL_object(df):
    i_star = 3004
    pca_DL = PCA(n_components=i_star)
    principalComponents_DL = pca_DL.fit_transform(df, df.shape)  
    return principalComponents_DL

pca_ml = PCA_ML_object(x_df)
pca_dl = PCA_DL_object(x_df)

def generate_ML_matrix(df, y):
    x = pca_ml.transform(df)
    x, y_with_jitter = add_jitter(x_pca_df, y, 0.01, 0.01)
    x = normalize(x)
    return x, y_with_jitter

def generate_DL_matrix(df, y):
    x = pca_dl.transform(df)
    x = normalize(x)
    return x. y
    