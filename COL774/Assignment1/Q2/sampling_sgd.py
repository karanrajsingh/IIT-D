# Imports - you can add any other permitted libraries
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import time

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

    #Generating X1 and X2 input vectors, and adding bias term to input
    X1 = np.random.normal(input_mean[0], input_sigma[0], N).reshape(N, 1)
    X2 = np.random.normal(input_mean[1], input_sigma[1], N).reshape(N,1)
    X_data = np.concatenate((X1, X2), axis = 1)
    X_intercept = np.ones([N, 1])
    X = np.concatenate((X_intercept, X_data), axis = 1)
    
    #Generating output vector y
    y_clean = np.matmul(X, theta)
    Noise = np.random.normal(0, noise_sigma, N)
    y = y_clean + Noise

    #Generating 80-20 train-test dataset split
    Xy = np.concatenate([X, y.reshape(y.shape[0], 1)], axis = 1)
    np.random.shuffle(Xy)
    Xy_train = Xy[ : math.floor(0.8*N)]
    Xy_test = Xy[math.floor(0.8*N) : ]
    y_train = Xy_train[ : , 3]
    X_train = Xy_train[ : , [1, 2]]
    y_test = Xy_test[ : , 3]
    X_test = Xy_test[ : , [1, 2]]
    
    return X_train, y_train, X_test, y_test

theta_list = []

class StochasticLinearRegressor:

    def __init__(self):
        ''' 
        Instructions:
        1. Toggle by commenting/uncommenting for selecting the batch size and threshold for training the dataset.
        2. For large batch sizes of 4 = 800000, choose 1e-10 as threshold.
        '''
        self.threshold = 1e-6
        #self.threshold = 1e-10
     
    
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
        X_intercept = np.ones([X.shape[0], 1])
        X = np.concatenate((X_intercept, X), axis = 1)

        # This gives the closed form solution
        XTX_inv = np.linalg.inv(np.matmul(X.T, X))
        XTy = np.matmul(X.T, y)
        Theta_mat = np.matmul(XTX_inv, XTy)
        print("Closed form solution: ", Theta_mat)

        arr = [1, 80, 8000, 800000]

        for i in range(len(arr)):
            batch_size = arr[i]

            #Starting parameters
            iteration = 0
            epoch = 0
            loss_str = 1
            loss_end = 0
            Theta = np.zeros(X.shape[1])
            batches = X.shape[0]//batch_size

            #Grouping data into batches
            Xy = np.concatenate([X, y.reshape(y.shape[0], 1)], axis = 1)
            np.random.shuffle(Xy)  
            Xy_data = Xy.reshape((batches,batch_size,Xy.shape[1]))
            global theta_list
            #Stochastic gradient descent algorithm
            t = np.array(Theta.reshape(1, X.shape[1]))
            if batch_size != 1:
                t1 = time.time()
                while abs(loss_end - loss_str) > self.threshold:
                    epoch_loss = 0
                    for batch in Xy_data:
                        iteration += 1
                        yb = batch[:,3]
                        xb = batch[:,[0,1,2]]
                        eta = np.dot(xb, Theta)
                        diff = yb - eta
                        grad = -1*np.matmul(xb.T, diff)/xb.shape[0]
                        Theta = Theta - learning_rate*grad
                        norm = np.linalg.norm(grad)
                        loss_diff = yb - np.dot(xb, Theta)
                        epoch_loss += np.matmul(loss_diff.T, loss_diff)/(2*xb.shape[0])
                        tt = Theta.reshape(1, Theta.shape[0])
                        t = np.append(t, tt, axis = 0)
                    epoch += 1
                    loss_str = loss_end
                    loss_end = epoch_loss     
                t2 = time.time()
                print('Time taken by SGD for {} batch size: '.format(batch_size), t2 - t1)
                print('Epochs: ', epoch)
                theta_list.append(t)
            else:
                flag = False
                t1 = time.time()
                while abs(loss_end - loss_str) > self.threshold:
                    epoch_loss = 0
                    for batch in Xy_data:
                        iteration += 1
                        yb = batch[:,3]
                        xb = batch[:,[0,1,2]]
                        eta = np.dot(xb, Theta)
                        diff = yb - eta
                        grad = -1*np.matmul(xb.T, diff)/xb.shape[0]
                        Theta = Theta - learning_rate*grad
                        norm = np.linalg.norm(grad)
                        loss_diff = yb - np.dot(xb, Theta)
                        epoch_loss += np.matmul(loss_diff.T, loss_diff)/(2*xb.shape[0])
                        tt = Theta.reshape(1, Theta.shape[0])
                        t = np.append(t, tt, axis = 0)
                        t1_ = time.time()
                        if t1_ - t1 > 1200:
                            flag = True
                            break
                    epoch += 1
                    loss_str = loss_end
                    loss_end = epoch_loss
                    if flag:
                        break
                t2 = time.time()
                print('Time taken by SGD for {} batch size: '.format(batch_size), t2 - t1)
                print('Epochs: ', epoch)
                theta_list.append(t)                
        return theta_list
    
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
        X_intercept = np.ones([X.shape[0], 1])
        X = np.concatenate((X_intercept, X), axis = 1)  
        y_predicted = []
        for i in range(len(theta_list)):
            param = theta_list[i][theta_list[i].shape[0] - 1]
            y_predicted.append(np.matmul(X, param))
        return y_predicted

    def error(self, y1, y2):
        #Adding bias term
        error_norm = []
        for i in range(len(y1)):
            error = y1[i] - y2
            val = np.square(np.linalg.norm(error))/(y2.shape[0])
            error_norm.append(val)
        return error_norm
    
    def plot(self, X, y):
        X_intercept = np.ones([X.shape[0], 1])
        X = np.concatenate((X_intercept, X), axis = 1)
        #Plotting movement of parameters
        global theta_list
        for i in range(len(theta_list)):
            fig = plt.figure()
            axes = fig.add_subplot(title = 'Movement of parameters with epochs, for batch size', projection = '3d', xlabel='theta0', ylabel='theta1', zlabel='  theta2')
            axes.scatter(theta_list[i][: , 0], theta_list[i][: , 1], theta_list[i][: , 2])      
            plt.show()

StochasticLinearRegressor()