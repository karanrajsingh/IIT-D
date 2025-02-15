# Imports - you can add any other permitted libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as anm

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

        self.t, self.theta = self.fit(X, y)
        print(len(self.theta))
        print(self.theta[len(self.theta) - 1])
        print(self.t[len(self.t) - 1])
        #print(self.t)

        print(self.predict(X[: 10]))

        self.plot(X, y)

    def readData(self, path_X, path_Y):
        X = np.loadtxt(path_X)
        y = np.loadtxt(path_Y)
        return X, y 

    def checkLoss(self, X, y, Theta):
        Hyp = np.matmul(X, Theta)
        L = y - Hyp
        loss = np.dot(L, L)/(2*X.shape[0])
        loss = np.square(np.linalg.norm(L))/(2*X.shape[0])
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
        Theta = np.zeros(X.shape[1])
        norm = 1e+12 
        #stopping criteria
        tolerance = 1e-12
        epochs = 0
        theta = []
        theta.append([epochs, Theta])
        t = np.array(Theta.reshape(1, X.shape[1]))
        print(t.shape, Theta.shape, t)

        while(norm > tolerance):
            epochs += 1
            A = np.dot(X, Theta)
            B = y - A
            Grad = -1*np.matmul(X.T, B)/X.shape[0]
            Theta = Theta - learning_rate*Grad
            norm = np.linalg.norm(Grad)
            theta.append([epochs, Theta])
            tt = Theta.reshape(1, Theta.shape[0])
            t = np.append(t, tt, axis = 0)
        self.theta = theta
        self.t = t
        return t, theta

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
        p = self.t[len(self.theta) - 1]
        y_predicted = np.matmul(X, param)
        print(X.shape, param.shape)
        print(X.shape, p.shape)
        y_p =  np.matmul(X, p)
        return y_p, y_predicted
    
    def plot(self, X, y):
        fig = plt.figure(figsize=(15, 9), facecolor='white')
        axes = plt.axes()
        plt.title('Linear Regression to predict wine density based on acidity')
        plt.scatter(X, y,s = 0.75,label = 'Training Data', c = 'blue')
        plt.xlabel('Wine Acidity')
        plt.ylabel('Wine Density')
        Ones = np.ones([X.shape[0],1])
        X = np.concatenate((Ones, X.reshape(X.shape[0], 1)), axis=1)
        #param = self.theta[len(self.theta) - 1][1]
        param = self.t[len(self.t) - 1]
        predicted = np.matmul(X, param)
        plt.plot(X[:,1], predicted, linewidth = 1.5, label = 'Learned Hypothesis', c = 'red')
        plt.legend(loc = "lower right")
        #plt.show()
        #plt.savefig('2.jpg')
        #plt.close()

        #Plotting Loss function
        d1 = np.linspace(-25, 50, num=300, endpoint=False)
        d2 = np.linspace(-10, 50, num=300, endpoint=False)
        v1, v2 = np.meshgrid(d1, d2)
        t  = np.stack([v1, v2], axis = -1)
        loss = np.apply_along_axis(lambda th : self.checkLoss(X, y, th), 2, t)
        fig = plt.figure()        
        axes2 = fig.add_subplot(title='Loss function', projection='3d', xlabel='theta0', ylabel='theta1', zlabel='  Loss')
        axes2.set_zlim3d(np.min(loss), np.max(loss))
        surface = axes2.plot_surface(v1, v2, loss, label='Loss Function', cmap = cm.gist_rainbow)        
        #plt.show()
        #plt.savefig('3a.jpg')
        #plt.close()

        #Plotting movement of parameters
        fig = plt.figure()
        axes3 = fig.add_subplot(title = 'Movement of parameters with epochs', projection = '3d', xlabel='theta0', ylabel='theta1', zlabel='  Loss')
        axes3.set_zlim3d(np.min(loss), 2*np.max(loss))
        surface = axes3.plot_surface(v1, v2, loss, label = 'Loss function', alpha = 0.25, cmap = cm.gist_rainbow)
        for i in range(len(self.theta) - 1):
            th_str = self.theta[i][1]
            th_end = self.theta[i+1][1]
            l_str = self.checkLoss(X, y, th_str)
            l_end= self.checkLoss(X, y, th_end)
            u = th_end[0] - th_str[0]
            v = th_end[1] - th_str[1]
            w = l_end - l_str
            axes3.quiver(th_str[0], th_str[1], l_str, u, v, w, colors = 'blue')       
        #plt.show()
        #plt.savefig('3b.jpg')
        #plt.close()

        #Plotting contours
        fig = plt.figure()
        axes4 = fig.add_subplot(title = 'Contour plot',  xlabel = 'theta0', ylabel = 'theta1')
        axes4.contour(v1, v2, loss, 100)

        #for i in range(len(self.t) - 1):
            #axes4.scatter(self.t[i][0], self.t[i][1])        
        axes4.scatter(self.t[:, 0], self.t[:, 1])

       
        plt.show()

LinearRegressor()