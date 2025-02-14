# Imports - you can add any other permitted libraries
import numpy as np
import matplotlib.pyplot as plt

# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.

class LogisticRegressor:
    
    # Assume Binary Classification
    def __init__(self):
        theta = []
        X, y = self.readAndNormalizeData()
        self.theta = self.fit(X, y, 0.1)

        self.plot(X, y)

    def readAndNormalizeData(self):
        #Read inputs from csv
        X = np.loadtxt('./../data/Q3/logisticX.csv', delimiter=',')
        X.reshape([X.shape[0], 2])

        #Normalize inputs
        Mean = np.mean(X, axis = 0)
        Var = np.var(X, axis = 0)
        Sigma = np.sqrt(Var)
        M = np.ones([X.shape[0], X.shape[1]])*Mean
        X = X - M
        X = np.divide(X, Sigma)

        # Read labels from csv 
        Y = np.loadtxt('./../data/Q3/logisticY.csv')
        #Y.reshape([Y.shape[0], 1])

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

        # Add intercept term to inputs
        Ones = np.ones([X.shape[0], 1])
        X = np.concatenate((Ones, X), axis = 1)
        Theta = np.zeros(X.shape[1], ) + 1e-10*np.ones(X.shape[1])

        norm = 1e+12
        tolerance = 1e-12
        epochs = 0
        theta = []
        theta.append([epochs, Theta])
        
        while(norm > tolerance):
            epochs += 1
            sigmoid = 1/(1+np.exp(-np.matmul(X, Theta)))
            difference = y - sigmoid
            grad = np.matmul(X.T, difference)
            differential = sigmoid*(1-sigmoid)
            diagonalize = np.diag(differential)
            hessian = np.matmul(-np.matmul(X.T, diagonalize), X)
            hessian_inv = np.linalg.inv(hessian)
            Theta = Theta - learning_rate*np.matmul(hessian_inv, grad)
            norm = np.linalg.norm(grad)
            theta.append([epochs, Theta])

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
            The predicted target label.
        """
        # Add intercept term to inputs
        Ones = np.ones([X.shape[0], 1])      
        X = np.concatenate((Ones, X), axis = 1)
        param = self.theta[len(self.theta) - 1][1]
        eta = np.matmul(X, param)
        sigmoid = 1/(1 + np.exp(-eta))
        y_predicted = np.zeros(X.shape[0])
        for i in range(len(y_predicted)):
            if sigmoid[i] >=0.5:
                y_predicted[i] = 1
        return y_predicted

    def plot(self, X, y):
        fig = plt.figure(figsize=(15, 9), facecolor='white')
        axes = plt.axes()
        plt.title("Logistic Regression using Newton's method")
        yX = np.concatenate((y.reshape(y.shape[0], 1), X), axis = 1)
        yX_class_1 = yX[yX[:,0] == 1]
        yX_class_0 = yX[yX[:,0] == 0]
        plt.scatter(yX_class_1[:,1], yX_class_1[:,2], s=30, label = 'Class 1', c ='green')
        plt.scatter(yX_class_0[:,1], yX_class_0[:,2], s=30, label = 'Class 0', c ='cyan')
        plt.xlabel('x1')
        plt.ylabel('x2')
        param = self.theta[len(self.theta) - 1][1] 
        x1_ = np.linspace(-3, 3, num = 50, endpoint=False)
        x2_ = -(param[1]/param[2])*x1_ - (param[0]/param[2])
        plt.plot(x1_, x2_, linewidth = 1.5, label = 'Decision boundary', c ='red')
        plt.legend(loc = "lower right")       
        plt.show()
        plt.savefig('logistic regression.jpg')

LogisticRegressor()