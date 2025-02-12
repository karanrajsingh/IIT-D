# Imports - you can add any other permitted libraries
import numpy as np

# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.

class LogisticRegressor:
    
    # Assume Binary Classification
    def __init__(self):
        X, y = self.readAndNormalizeData()
        Theta = self.fit(X, y, 0.1)
        self.theta = Theta
        print(self.theta)
        print('Class: ',self.predict([1, 4.2903,2.7389]))

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
        X = np.concatenate((Ones, X1), axis = 1)

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
        Theta = np.zeros(X.shape[1], ) + 1e-10*np.ones(X.shape[1])

        norm = 1e+12
        tolerance = 1e-12
        
        while(norm > tolerance):

            sigmoid = 1/(1+np.exp(-np.matmul(X, Theta)))
            difference = y - sigmoid
            grad = np.matmul(X.T, difference)
            differential = sigmoid*(1-sigmoid)
            diagonalize = np.diag(differential)
            hessian = np.matmul(-np.matmul(X.T, diagonalize), X)
            hessian_inv = np.linalg.inv(hessian)
            Theta = Theta - learning_rate*np.matmul(hessian_inv, grad)
            norm = np.linalg.norm(grad)

        return Theta
    
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
        eta = np.dot(X, self.theta)
        sigmoid = 1/(1 + np.exp(-eta))
        if sigmoid >=0.5:
            return 1
        else:
            return 0

LogisticRegressor()