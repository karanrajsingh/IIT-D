# Imports - you can add any other permitted libraries
import numpy as np
import math
import matplotlib.pyplot as plt

# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.


class GaussianDiscriminantAnalysis:
    # Assume Binary Classification
    def __init__(self):
        self.X, self.y, self.Mean, self.Sigma = self.readAndNormalizeData()
        self.same_Cov_tuple = self.fit(self.X, self.y, True)
        self.same_Cov_Mean_0 = self.same_Cov_tuple[0]
        self.same_Cov_Mean_1 = self.same_Cov_tuple[1]
        self.same_Cov = self.same_Cov_tuple[2]
        print('init - Same Cov: ',self.same_Cov_Mean_0, self.same_Cov_Mean_1, self.same_Cov)
        self.diff_Cov_tuple = self.fit(self.X, self.y, False)
        self.diff_Cov_Mean_0 = self.diff_Cov_tuple[0]
        self.diff_Cov_Mean_1 = self.diff_Cov_tuple[1]
        self.diff_Cov_0 = self.diff_Cov_tuple[2]
        self.diff_Cov_1 = self.diff_Cov_tuple[3]
        print('diff - Same Cov: ',self.diff_Cov_Mean_0, self.diff_Cov_Mean_1, self.diff_Cov_0, self.diff_Cov_1)

        #self.plot(self.X, self.y)
    
    def normalizeData(self, X):
        X.reshape([X.shape[0], 2])
        #Normalize inputs
        Mean = np.mean(X, axis = 0)
        Var = np.var(X, axis = 0)
        Sigma = np.sqrt(Var)
        M = np.ones([X.shape[0], X.shape[1]])*Mean
        X = X - M
        X = np.divide(X, Sigma)
        return X, Mean, Sigma

    def readAndNormalizeData(self):
        #Read inputs from csv
        X1 = np.loadtxt('./../data/Q4/q4x.dat')

        X, Mean, Sigma = self.normalizeData(X1)

        # Read labels from csv 
        Y = np.zeros(X.shape[0])
        with open('./../data/Q4/q4y.dat', 'r') as f:
            data = f.readlines()
        for i in range(len(data)):
            if data[i] == 'Alaska\n':
                Y[i] = 1
            else:
                Y[i] = 0
        
        return X1, Y, Mean, Sigma

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
        X, Mean, Sigma = self.normalizeData(X)

        phi = np.sum(y, axis = 0)/y.shape[0]

        X_0 = X[np.where(y==0)]
        Mean_0 = np.mean(X_0, axis = 0)
        X_1 = X[np.where(y==1)]
        Mean_1 = np.mean(X_1, axis = 0)
        
        if assume_same_covariance:
            Ones_0 = np.ones([X_0.shape[0], X_0.shape[1]])
            Ones_1 = np.ones([X_1.shape[0], X_1.shape[1]])
            Mean_0_Vec = Ones_0*Mean_0
            Mean_1_Vec = Ones_1*Mean_1
            merge_mean =  np.concat((Mean_0_Vec, Mean_1_Vec), axis=0)
            merge_X = np.concat((X_0, X_1), axis=0)
            diff = merge_X - merge_mean
            Cov = np.matmul(diff.T, diff)/diff.shape[0]
            same_slope = np.matmul(np.linalg.inv(Cov), (Mean_1 - Mean_0))
            same_intercept = math.log(phi/(1 - phi)) - 0.5*(np.matmul(np.matmul(Mean_1.T, np.linalg.inv(Cov)), Mean_1) - np.matmul(np.matmul(Mean_0.T, np.linalg.inv(Cov)), Mean_0))
            print('Param - Same Cov', same_slope, same_intercept)
            return (Mean_0, Mean_1, Cov)
        else:
            diff_0 = X_0 - Mean_0
            Cov_0 = np.matmul(diff_0.T, diff_0)/diff_0.shape[0]
            diff_1 = X_1 - Mean_1
            Cov_1 = np.matmul(diff_1.T, diff_1)/diff_1.shape[0]
            second_order = np.linalg.inv(Cov_0) - np.linalg.inv(Cov_1)
            same_slope = np.matmul(np.linalg.inv(Cov_1), Mean_1) - np.matmul(np.linalg.inv(Cov_0), Mean_0)
            same_intercept = math.log(phi/(1 - phi)) + 0.5*math.log(np.linalg.norm(Cov_0)/np.linalg.norm(Cov_1)) - 0.5*(np.matmul(np.matmul(Mean_1.T, np.linalg.inv(Cov_1)), Mean_1) - np.matmul(np.matmul(Mean_0.T, np.linalg.inv(Cov_0)), Mean_0))
            print('Param - Diff Cov', same_slope,same_intercept, second_order)
            return (Mean_0, Mean_1, Cov_0, Cov_1)

    def param(self, y,assume_same_covariance):
        phi = np.sum(y, axis = 0)/y.shape[0]
        if assume_same_covariance:
            Mean_0 = self.same_Cov_tuple[0]
            Mean_1 = self.same_Cov_tuple[1]
            Cov = self.same_Cov_tuple[2]
            same_slope = np.matmul(np.linalg.inv(Cov), (Mean_1 - Mean_0))
            same_intercept = math.log(phi/(1 - phi)) - 0.5*(np.matmul(np.matmul(Mean_1.T, np.linalg.inv(Cov)), Mean_1) - np.matmul(np.matmul(Mean_0.T, np.linalg.inv(Cov)), Mean_0))            
            return (same_intercept, same_slope)
        else:
            Mean_0 = self.diff_Cov_tuple[0]
            Mean_1 = self.diff_Cov_tuple[1]
            Cov_0 = self.diff_Cov_tuple[2]
            Cov_1 = self.diff_Cov_tuple[3]
            second_order = 0.5*(np.linalg.inv(Cov_0) - np.linalg.inv(Cov_1))
            diff_slope = np.matmul(np.linalg.inv(Cov_1), Mean_1) - np.matmul(np.linalg.inv(Cov_0), Mean_0)
            diff_intercept = math.log(phi/(1 - phi)) + 0.5*math.log(np.linalg.det(Cov_0)/np.linalg.det(Cov_1)) - 0.5*(np.matmul(np.matmul(Mean_1.T, np.linalg.inv(Cov_1)), Mean_1) - np.matmul(np.matmul(Mean_0.T, np.linalg.inv(Cov_0)), Mean_0))
            return (diff_intercept, diff_slope, second_order)

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
        #Normalize inputs first
        X.reshape([X.shape[0], 2])
        #Normalize inputs w.r.t training mean and variance
        M = np.ones([X.shape[0], X.shape[1]])*self.Mean
        X = X - M
        X = np.divide(X, self.Sigma)
        intercept, slope = self.param(self.y, True)
        val = intercept + np.matmul(X, slope)
        y_p = np.zeros(val.shape[0])
        for i in range(val.shape[0]):
            if val[i] >= 0:
                y_p[i] = 1
        return y_p

    def plot(self, X, y):
        X, Mean, Sigma = self.normalizeData(X)
        fig = plt.figure(figsize=(15, 9), facecolor='white')
        axes = plt.axes()
        plt.title("Gaussian Discriminant Analysis")
        yX = np.concatenate((y.reshape(y.shape[0], 1), X), axis = 1)
        yX_class_1 = yX[yX[:,0] == 1]
        yX_class_0 = yX[yX[:,0] == 0]
        plt.scatter(yX_class_1[:,1], yX_class_1[:,2], s=30, label = 'Alaska', c ='cyan')
        plt.scatter(yX_class_0[:,1], yX_class_0[:,2], s=30, label = 'Canada', c ='green')
        plt.xlabel('Ring diameter in fresh water')
        plt.ylabel('Ring diameter in marine water')
        same_param = self.param(y, True)
        diff_param = self.param(y, False)
        print('same param', same_param)
        print('diff param',diff_param)
        x1_ = np.linspace(-3, 3, num = 50, endpoint=False)
        print(same_param[0], same_param[1][0], same_param[1][1])
        x2_ = -(same_param[1][0]/same_param[1][1])*x1_ - (same_param[0]/same_param[1][1])
        plt.plot(x1_, x2_, linewidth = 1.5, label = 'Linear decision boundary', c ='red')
        x1_diff = np.linspace(-3, 3, num = 50, endpoint=False)
        x2_diff = np.linspace(-3, 3, num= 50, endpoint= False)
        v1, v2 = np.meshgrid(x1_diff, x2_diff)
        v = np.stack([v1, v2], axis = -1)
        print(diff_param)
        z = np.apply_along_axis(lambda x: self.strips(x, diff_param), 2, v)
        axes.contour(v1, v2, z, [0], colors='magenta')
        axes.plot([], [], color='magenta', label='Quadratic decision boundary') 
        plt.legend(loc = "lower right")       
        plt.show()

    def strips(self, x, diff_param):
        z = diff_param[0] + np.matmul(x.T, diff_param[1]) + np.matmul(np.matmul(x.T, diff_param[2]) , x)
        return z

#GaussianDiscriminantAnalysis()