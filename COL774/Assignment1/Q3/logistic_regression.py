# Imports - you can add any other permitted libraries
import numpy as np

# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.

class LogisticRegressor:
    # Assume Binary Classification
    def __init__(self):
        X, y = self.readAndNormalizeData()

        print(X, y)


    def readAndNormalizeData(self):
        #Read inputs from csv
        X1 = np.loadtxt('./../data/Q3/logisticX.csv', delimiter=',')
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
        X = np.concatenate((X1, Ones), axis = 1)

        # Read labels from csv 
        Y = np.loadtxt('./../data/Q3/logisticY.csv')
        Y.reshape([Y.shape[0], 1])

        return X, Y

    
    def fit(self, X, y, learning_rate=0.01):
        """
        Fit the linear regression model to the data using Newton's Method.
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
        List of Parameters: numpy array of shape (n_iter, n_features+1,)
            The list of parameters obtained after each iteration of Newton's Method.
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

LogisticRegressor()