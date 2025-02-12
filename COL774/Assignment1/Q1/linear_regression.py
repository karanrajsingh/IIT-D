# Imports - you can add any other permitted libraries
import numpy as np

# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.

class LinearRegressor:
    def __init__(self):

        X, y = self.readData()
        
        Theta = self.fit(X, y)
        print(Theta)
        self.theta = Theta

        loss = self.checkLoss(X, y, Theta)        

    def readData(self):

        X = self.readDataX()
        y = self.readDataY()
        
        return X, y
    
    def readDataX(self):
        '''
        Read X from linearX.csv and create numpy array X

        ''' 
        D = np.loadtxt('./../data/Q1/linearX.csv', ndmin = 2)
        A = np.ones([D.shape[0],1])
        X = np.concatenate((A, D), axis=1)
        #print(X.shape)
        return X
    
    def readDataY(self):
        '''
        Read Y from linearY.csv and create numpy array Y
        
        '''
        y = np.loadtxt('./../data/Q1/linearY.csv')
        #print(y.shape)
        return y

    def checkLoss(self, X, y, Theta):
        #Param = np.array([29.06475515, 6.21867801])
        #Hyp = np.matmul(X, Param)
        Hyp = np.matmul(X, Theta)
        L = y - Hyp
        loss = np.dot(L, L)/(2*X.shape[0])
        return loss       

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
        

        #Starting parameters
        Theta = np.zeros(2)
        norm = 1e+12 
        #stopping criteria
        tolerance = 1e-12
        
        while(norm > tolerance):
            
            A = np.dot(X, Theta)
            B = y - A
            Grad = -1*np.matmul(X.T, B)/X.shape[0]
            Theta = Theta - learning_rate*Grad
            norm = np.linalg.norm(Grad)

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
            The predicted target values.
        """
        val = np.dot(X, self.theta)
        return val

LinearRegressor()