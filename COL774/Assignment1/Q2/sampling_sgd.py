# Imports - you can add any other permitted libraries
import numpy as np
import math

# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.

def generate(N, theta, input_mean, input_sigma, noise_sigma):
    """
    Generate normally distributed input data and target values
    Note that we have 2 input features
    Parameters
    ----------
    N : int
        The number of samples to generate.
        
    theta : numpy array of shape (3,)
        The true parameters of the linear regression model.
        
    input_mean : numpy array of shape (2,)
        The mean of the input data.
        
    input_sigma : numpy array of shape (2,)
        The standard deviation of the input data.
        
    noise_sigma : float
        The standard deviation of the Gaussian noise.
        
    Returns
    -------
    X : numpy array of shape (N, 2)
        The input data.
        
    y : numpy array of shape (N,)
        The target values.
    """

    '''

    #Generate using covariance. This works!
    Cov = np.eye(input_sigma.shape[0])
    for i in range(input_sigma[0]):
        Cov[i][i] = input_sigma[i]

    X_data =  np.random.multivariate_normal(input_mean, Cov, N)
    X_intercept = np.ones([X_data.shape[0], 1])
    X = np.concatenate((X_intercept, X_data), axis = 1)

    '''

    #As both x1 and x2 are assumed to be independent

    X1 = np.random.normal(input_mean[0], input_sigma[0], N).reshape(N, 1)
    X2 = np.random.normal(input_mean[1], input_sigma[1], N).reshape(N,1)
    X_data = np.concatenate((X1, X2), axis = 1)
    X_intercept = np.ones([N, 1])
    X = np.concatenate((X_intercept, X_data), axis = 1)
    print(X1[1], X2[1], X_data[1], X[1])

    #print(X1)
    #print(X2)
    #print(X_data)
    #print(X)

    

class StochasticLinearRegressor:
    def __init__(self):
        N = int(1e6)
        theta = np.array([3, 1, 2])
        #Correct shape
        #print(theta.shape)
        input_mean = np.array([3, -1])
        input_sigma = np.array([2, 2])
        #Correct shape
        #print(input_mean.shape, input_sigma.shape)
        noise_sigma = math.sqrt(2)
        generate(N, theta, input_mean, input_sigma, noise_sigma)


    
    def fit(self, X, y, learning_rate=0.01):
        """
        Fit the linear regression model to the data using Gradient Descent.
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input data.
            
        y : numpy array of shape (n_samples,)
            The target values.

        learning_rate : float
            The learning rate to use in the update rule.
            
        Returns
        -------
        List of Parameters: numpy array of shape (n_iter, n_features+1,)
            The list of parameters obtained after each iteration of Gradient Descent.
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
            The predicted target values.
        """
        pass

StochasticLinearRegressor()