# Imports - you can add any other permitted libraries
import numpy as np
import math

# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.


class GaussianDiscriminantAnalysis:
    # Assume Binary Classification
    def __init__(self):
        X, y = self.readAndNormalizeData()
        #print(y)
        self.fit(X, y)
    
    def readAndNormalizeData(self):
        #Read inputs from csv
        X1 = np.loadtxt('./../data/Q4/q4x.dat')
        X1.reshape([X1.shape[0], 2])

        #Normalize inputs
        Mean = np.mean(X1, axis = 0)
        Var = np.var(X1, axis = 0)
        Sigma = np.sqrt(Var)
        M = np.ones([X1.shape[0], X1.shape[1]])*Mean
        X1 = X1 - M
        X1 = np.divide(X1, Sigma)
        X = X1

        # Add intercept term to inputs
        #Ones = np.ones([X1.shape[0], 1])
        #X = np.concatenate((Ones, X1), axis = 1)

        # Read labels from csv 
        Y = np.zeros(X.shape[0])
        with open('./../data/Q4/q4y.dat', 'r') as f:
            data = f.readlines()
        for i in range(len(data)):
            if data[i] == 'Alaska\n':
                Y[i] = 1
            else:
                Y[i] = 0
        
        return X, Y

    def fit(self, X, y, assume_same_covariance=False):
        """
        Fit the Gaussian Discriminant Analysis model to the data.
        Remember to normalize the input data X before fitting the model.
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input data.
            
        y : numpy array of shape (n_samples,)
            The target labels - 0 or 1.
        
        learning_rate : float
            The learning rate to use in the update rule.
        
        Returns
        -------
        Parameters: 
            If assume_same_covariance = True - 3-tuple of numpy arrays mu_0, mu_1, sigma 
            If assume_same_covariance = False - 4-tuple of numpy arrays mu_0, mu_1, sigma_0, sigma_1
            The parameters learned by the model.
        """
        phi = np.sum(y, axis = 0)/y.shape[0]

        X_0 = X[np.where(y==0)]
        Mean_0 = np.mean(X_0, axis = 0)
        X_1 = X[np.where(y==1)]
        Mean_1 = np.mean(X_1, axis = 0)
        
        Ones_0 = np.ones([X_0.shape[0], X_0.shape[1]])
        Ones_1 = np.ones([X_1.shape[0], X_1.shape[1]])
        Mean_0_Vec = Ones_0*Mean_0
        Mean_1_Vec = Ones_1*Mean_1
        merge_mean =  np.concat((Mean_0_Vec, Mean_1_Vec), axis=0)
        merge_X = np.concat((X_0, X_1), axis=0)
        diff = merge_X - merge_mean
        Cov = np.matmul(diff.T, diff)/diff.shape[0]
        same_slope = np.matmul(np.linalg.inv(Cov), (Mean_1 - Mean_0))
        same_intercept = math.log(phi/(1 - phi)) - 0.5*(np.matmul(np.matmul(Mean_1.T, np.linalg.inv(Cov)), Mean_1) - np.matmul(np.matmul(Mean_0.T, np.linalg.inv(Cov)), Mean_0))
        print(same_slope)
        print(same_intercept)

        diff_0 = X_0 - Mean_0
        Cov_0 = np.matmul(diff_0.T, diff_0)/diff_0.shape[0]
        diff_1 = X_1 - Mean_1
        Cov_1 = np.matmul(diff_1.T, diff_1)/diff_1.shape[0]
        second_order = np.linalg.inv(Cov_0) - np.linalg.inv(Cov_1)
        same_slope = np.matmul(np.linalg.inv(Cov_1), Mean_1) - np.matmul(np.linalg.inv(Cov_0), Mean_0)
        same_intercept = math.log(phi/(1 - phi)) + 0.5*math.log(np.linalg.norm(Cov_0)/np.linalg.norm(Cov_1)) - 0.5*(np.matmul(np.matmul(Mean_1.T, np.linalg.inv(Cov)), Mean_1) - np.matmul(np.matmul(Mean_0.T, np.linalg.inv(Cov)), Mean_0))
        print(second_order)
        print(same_slope)
        print(same_intercept)


    def predict(self, X):
        """
        Predict the target values for the input data.
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input data.
            
        Returns
        -------
        y_pred : numpy array of shape (n_samples,)
            The predicted target label.
        """
        pass
GaussianDiscriminantAnalysis()