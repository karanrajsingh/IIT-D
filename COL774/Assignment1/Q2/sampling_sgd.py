# Imports - you can add any other permitted libraries
import numpy as np
import matplotlib.pyplot as plt
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

    #As both x1 and x2 are assumed to be independent
    X1 = np.random.normal(input_mean[0], input_sigma[0], N).reshape(N, 1)
    X2 = np.random.normal(input_mean[1], input_sigma[1], N).reshape(N,1)
    X_data = np.concatenate((X1, X2), axis = 1)
    X_intercept = np.ones([N, 1])
    X = np.concatenate((X_intercept, X_data), axis = 1)
    
    y_clean = np.matmul(X, theta)
    Noise = np.random.normal(0, noise_sigma, N)
    y = y_clean + Noise

    Xy = np.concatenate([X, y.reshape(y.shape[0], 1)], axis = 1)
    np.random.shuffle(Xy)
    Xy_train = Xy[ : math.floor(0.8*N)]
    Xy_test = Xy[math.floor(0.8*N) : ]
    y_train = Xy_train[ : , 3]
    X_train = Xy_train[ : , [0, 1, 2]]
    y_test = Xy_test[ : , 3]
    X_test = Xy_test[ : , [0, 1, 2]]
    
    return X_train, y_train, X_test, y_test

class StochasticLinearRegressor:
    def __init__(self):
        N = int(1e6)
        theta = np.array([3, 1, 2])
        input_mean = np.array([3, -1])
        input_sigma = np.array([2, 2])
        noise_sigma = math.sqrt(2)
        X_train, y_train, X_test, y_test = generate(N, theta, input_mean, input_sigma, noise_sigma)
        
        self.fit(X_train, y_train)
    
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
        iteration = 0
        epoch = 0
        Theta = np.zeros(X.shape[1])
        norm = 1e+12 
        #stopping criteria
        tolerance = 1e-6
        batch_size = 800000
        batches = X.shape[0]//batch_size
        
        Xy = np.concatenate([X, y.reshape(y.shape[0], 1)], axis = 1)
        np.random.shuffle(Xy)  
        Xy_data = Xy.reshape((batches,batch_size,Xy.shape[1]))
        #print(Xy_data)

        XTX_inv = np.linalg.inv(np.matmul(X.T, X))
        XTy = np.matmul(X.T, y)
        Theta_mat = np.matmul(XTX_inv, XTy)
        print(Theta_mat)

        #Change this
        curr_loss = 1
        prev_loss = 0

        while abs(curr_loss - prev_loss) > tolerance:
            epoch_loss = 0
            for batch in Xy_data:
                iteration += 1
                #print(batch.shape)
                yb = batch[:,3]
                #print('yb ',yb.shape)
                xb = batch[:,[0,1,2]]
                #print('xb ', xb.shape)
                #print('Theta ',Theta.shape)
                eta = np.dot(xb, Theta)
                #print('eta ',eta.shape)
                diff = yb - eta
                #print('diff ', diff.shape)
                grad = -1*np.matmul(xb.T, diff)/xb.shape[0]
                #print('grad ', grad)
                Theta = Theta - 0.001*grad
                #print('Theta ', Theta)
                norm = np.linalg.norm(grad)
                #print('grad norm ', norm)
                loss_diff = yb - np.dot(xb, Theta)
                epoch_loss += np.matmul(loss_diff.T, loss_diff)/(2*xb.shape[0])
            epoch += 1
            prev_loss = curr_loss
            curr_loss = epoch_loss
            print(epoch, curr_loss, prev_loss)
            


        '''
        while norm > tolerance:
            for batch in Xy_data:
                iteration += 1
                batch = Xy_data
                Xe = batch[:,[0,1,2]]
                print(Xe.shape)
                ye = batch[:,3]
                print(ye.shape)
                
                eta = np.dot(Xe, Theta)
                diff = ye - eta
                Grad = -np.matmul(Xe.T, diff)/Xe.shape[0]
                Theta = Theta - learning_rate*Grad
                norm = np.linalg.norm(Grad)
                if norm <= tolerance:
                    break

                
                
            
            epoch += 1

        '''

        print(Theta)

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
        pass

StochasticLinearRegressor()