import cvxopt
import numpy as np
import cv2

class SupportVectorMachine:
    '''
    Binary Classifier using Support Vector Machine
    '''
    def __init__(self):
        path = "../data/Q2/train/dew/2208.jpg"
        image = cv2.imread(path)
        # print(image.shape)
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        b, g, r = cv2.split(image)
        # cv2.imshow("blue", b)
        # cv2.waitKey(0)
        # cv2.imshow("green", g)
        # cv2.waitKey(0)
        # cv2.imshow("red", r)
        # cv2.waitKey(0)
        update = cv2.resize(image, (100, 100))
        # print(update.shape)
        # cv2.imshow("image", update)
        # cv2.waitKey(0)
        cropped = update[0:100, 0:100]
        # print(cropped.shape)
        # cv2.imshow("image", cropped)
        # cv2.waitKey(0)
        # print(r, g, b)

        flatten = cropped.flatten()
        # print(flatten)
        # print(type(flatten))
        normalized = np.multiply(flatten, 1/255)
        print(normalized)
        
    def fit(self, X, y, kernel = 'linear', C = 1.0, gamma = 0.001):
        '''
        Learn the parameters from the given training data
        Classes are 0 or 1
        
        Args:
            X: np.array of shape (N, D) 
                where N is the number of samples and D is the flattened dimension of each image
                
            y: np.array of shape (N,)
                where N is the number of samples and y[i] is the class of the ith sample
                
            kernel: str
                The kernel to be used. Can be 'linear' or 'gaussian'
                
            C: float
                The regularization parameter
                
            gamma: float
                The gamma parameter for gaussian kernel, ignored for linear kernel
        '''
        pass

    def predict(self, X):
        '''
        Predict the class of the input data
        
        Args:
            X: np.array of shape (N, D) 
                where N is the number of samples and D is the flattened dimension of each image
                
        Returns:
            np.array of shape (N,)
                where N is the number of samples and y[i] is the class of the
                ith sample (0 or 1)
        '''
        
        pass