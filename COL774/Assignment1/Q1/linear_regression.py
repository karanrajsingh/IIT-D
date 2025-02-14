# Imports - you can add any other permitted libraries
import numpy as np
import matplotlib.pyplot as plt

# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.

class LinearRegressor:
    def __init__(self):        
        self.theta = []
        #Read data from file
        path_X = './../data/Q1/linearX.csv'
        path_Y = './../data/Q1/linearY.csv'
        X, y = self.readData(path_X, path_Y)

        self.theta = self.fit(X, y)
        print(self.theta[len(self.theta) - 1])
        #print(self.predict(X[: 10]))

        self.plot(X, y)

    def readData(self, path_X, path_Y):
        X = np.loadtxt(path_X)
        y = np.loadtxt(path_Y)
        return X, y 

    def checkLoss(self, X, y, Theta):
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

        #Adding bias term
        Ones = np.ones([X.shape[0],1])
        X = np.concatenate((Ones, X.reshape(X.shape[0], 1)), axis=1)

        #Starting parameters
        Theta = np.zeros(2)
        norm = 1e+12 
        #stopping criteria
        tolerance = 1e-12
        epochs = 0
        theta = []
        theta.append([epochs, Theta])

        while(norm > tolerance):
            epochs += 1
            A = np.dot(X, Theta)
            B = y - A
            Grad = -1*np.matmul(X.T, B)/X.shape[0]
            Theta = Theta - learning_rate*Grad
            norm = np.linalg.norm(Grad)
            theta.append([epochs, Theta])
        self.theta = theta
        return theta

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
        
        #Adding bias term
        Ones = np.ones([X.shape[0],1])
        X = np.concatenate((Ones, X.reshape(X.shape[0], 1)), axis=1)
        param = self.theta[len(self.theta) - 1][1]
        y = np.matmul(X, param)
        return y
    
    def plot(self, X, y):
        fig = plt.figure(figsize=(15, 9), facecolor='white')
        axes = plt.axes()
        plt.title('Linear Regression to predict wine density based on acidity')
        plt.scatter(X, y,s = 15,label = 'Training Data', c = 'blue')
        plt.xlabel('Wine Acidity')
        plt.ylabel('Wine Density')
        Ones = np.ones([X.shape[0],1])
        X = np.concatenate((Ones, X.reshape(X.shape[0], 1)), axis=1)
        param = self.theta[len(self.theta) - 1][1] 
        predicted = np.matmul(X, param)
        plt.plot(X[:,1], predicted, linewidth = 2,label = 'Learned Hypothesis', c = 'red')
        plt.legend(loc = "lower right")
        plt.show()
        plt.savefig('linear regression.jpg')

LinearRegressor()