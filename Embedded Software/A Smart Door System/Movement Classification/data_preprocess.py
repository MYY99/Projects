# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 21:34:05 2022

@author: User
"""

import numpy as np

def clean_data(data, save_csv = None, fname_clean = 'cleaned.csv', fname_raw = 'raw.csv'):
    """clean dist. data
    :param data: dist. data to clean, numpy array of shape (num_sample, num_feature)
    :param save_csv: output csv, Boolean <True or False>
    :param filename: csv filename, string <filename.csv>
    :return cleaned data, number of cleaned samples
    """
    if save_csv:
        np.savetxt(fname_raw, data, delimiter=",")
        # np.savetxt(fname_clean, X, delimiter=",")
    
    X = data
    num_sample, num_fea = X.shape
    cleaned = 0
    
    for i in range(num_sample):
        if (X[i,:] > 200).any():
            cleaned += 1
            # print('before: ', i, X[i, :])
            for j in range(num_fea):
                if X[i, j] > 200:
                    if j == 0:
                        if X[i, j+1] > 200 and X[i, j+2] > 200:
                            X[i, j] = 200
                        else:
                            X[i, j] = np.min([X[i, j+1], X[i, j+2]])
                    elif (j > 0) and (j < (num_fea - 1)):
                        if X[i, j+1] > 200:
                            if j != num_fea - 2 and X[i, j+2] > 200:
                                X[i, j] = 200
                            else:
                                X[i, j] = X[i, j-1]
                        else:
                            X[i, j] = np.mean([X[i, j-1], X[i, j+1]])
                    elif j == num_fea - 1:
                        X[i, j] = X[i, j-1]
            # print('after: ', i, X[i, :])
            
    if save_csv:
        # np.savetxt(fname_raw, data, delimiter=",")
        np.savetxt(fname_clean, X, delimiter=",")
        
    return X, cleaned
                            