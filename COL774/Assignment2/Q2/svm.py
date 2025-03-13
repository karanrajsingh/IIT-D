import cvxopt
import numpy as np
import cv2
import os

class SupportVectorMachine:
    '''
    Binary Classifier using Support Vector Machine
    '''
    
    train = ""
    test = ""
    

    def __init__(self):
        self.read_images("../data/Q2/")
        self.print()

    def print(self):
        global train
        global test
        print("------Train global------")
        for array in train:
            print(array.shape)
        print("------Test global------")
        for array in test:    
            print(array.shape)
    
    def read_images(self, path):
        array_list_train = []
        array_list_test = []
        array_list_train_corrupted = []
        array_list_test_corrupted = []
        path_train = os.path.join(path, "train")
        path_test = os.path.join(path, "test")
        label_train = [label for label in os.listdir(path_train)]
        label_test = [label for label in os.listdir(path_test)]
        for label in label_train:
            path_train_label = os.path.join(path_train, label)
            X_train, X_train_corrupted = self.read_images_class(path_train_label)
            array_list_train.append(X_train)
            array_list_train_corrupted.append(X_train_corrupted)
        for label in label_test:
            path_test_label = os.path.join(path_test, label)
            X_test, X_test_corrupted = self.read_images_class(path_test_label)
            array_list_test.append(X_test)
            array_list_test_corrupted.append(X_test_corrupted)
        print(array_list_train_corrupted)
        print(array_list_test_corrupted)
        global train 
        train = array_list_train
        global test 
        test = array_list_test
        return array_list_train, array_list_test

    def read_images_class(self, path):
        array_list = []
        array_list_corrupted = []
        images = [image for image in os.listdir(path)]
        for image in images:
            image_path = os.path.join(path, image)
            image_read = cv2.imread(image_path)
            if image_read is not None:
                x = self.pre_process_image(image_read)
                array_list.append(x)
            else:
                array_list_corrupted.append(image_path)
        X = np.array(array_list).T
        return X, array_list_corrupted


    def pre_process_image(self, image):
        update = cv2.resize(image, (100, 100))
        cropped = update[0:100, 0:100]
        flatten = cropped.flatten()
        normalized = np.multiply(flatten, 1/255)
        return normalized

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
        # https://cvxopt.org/userguide/coneprog.html#quadratic-programming

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