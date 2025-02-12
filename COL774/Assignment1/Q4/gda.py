# Imports - you can add any other permitted libraries
import numpy as np

# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.


class GaussianDiscriminantAnalysis:
    # Assume Binary Classification
    def __init__(self):
        X, y = self.readAndNormalizeData()
        print(y)
    
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

        # Add intercept term to inputs
        Ones = np.ones([X1.shape[0], 1])
        X = np.concatenate((Ones, X1), axis = 1)

        # Read labels from csv 
        Y = np.zeros(X.shape[0])
        with open('./../data/Q4/q4y.dat', 'r') as f:
            data = f.readlines()
        for i in range(len(data)):
            if data[i] == 'Alaska\n':
                Y[i] = 1
            else:
                Y[i] = 0
        print(Y[49], Y[50]) 
        
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
        pass
    
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