import cvxopt
import numpy as np
import cv2
import os
import math
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

class SupportVectorMachine:
    '''
    Binary Classifier using Support Vector Machine
    '''
    
    train = ""
    test = ""
    w = ""
    b = ""
    support_vectors = ""
    X_supp = ""
    y_supp = ""

    def __init__(self):
        """
        1. Read images for all classes.
        2. Process them as: Resize to (100, 100, 3) -> centre crop -> flatten -> pixel normalization.
        3. Store images in global 'train' and 'test' variables, with the structure:
            var:: train/test:
                        class 'dew'         : X_var_dew
                        class 'fogsmog''    : X_var_dew
                        :
                        :
                        class 'i'           : X_var_i
                        :
                        :
                        class 'snow'        : X_var_snow
        """


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
        X = np.array(array_list)
        return X, array_list_corrupted

    def pre_process_image(self, image):
        update = cv2.resize(image, (100, 100))
        cropped = update[0:100, 0:100]
        flatten = cropped.flatten()
        normalized = np.multiply(flatten, 1/255)
        return normalized
    
    '''
    1. Builds inputs (X,y) for fit function for binary classfication.
    2. Concatenates the X_train_Class_0 and X_train_Class_1 numpy arrays.
    3. Concatenates the y_train_Class_0 and y_train_Class_1 numpy arrays.
        - Here y is (-1, 1)
    4. Repeats steps 2 & 3 for test sets.
    5. Returns these 4 values
    '''

    def build_input(self, class_index_0 = 0, class_index_1 = 1):
        global train
        X_train_0 = train[class_index_0]
        X_train_1 = train[class_index_1]
        X_train = np.vstack((X_train_0, X_train_1))
        m0 = X_train_0.shape[0]
        m1 = X_train_1.shape[0]
        y_train_0 = -1*np.ones(m0)
        y_train_1 = np.ones(m1)
        y_train = np.hstack((y_train_0, y_train_1))
        global test
        X_test_0 = test[class_index_0]
        X_test_1 = test[class_index_1]
        X_test = np.vstack((X_test_0, X_test_1))
        m0 = X_test_0.shape[0]
        m1 = X_test_1.shape[0]
        y_test_0 = -1*np.ones(m0)
        y_test_1 = np.ones(m1)
        y_test = np.hstack((y_test_0, y_test_1))
        return X_train, y_train, X_test, y_test
    
    """
    1. Analysis performance for multi-class classficiation for all classifiers:
    """

    def analyse(self):
        global train, test
        for i in range(len(train)):
            for j in range(i+1, len(train)):
                print("Training for classes: {0} and {1}".format(i, j))
                X_train, y_train, X_test, y_test = self.build_input(class_index_0 = i, class_index_1= j)
                self.fit(X_train, y_train, kernel='gaussian')
                prediction = self.predict_gaussian(X_test)
                accuracy = self.accuracy(y_test, prediction)
                # self.sort_and_plot(X_train)

    def analyse_svc(self, kernel='linear'):
        global train, test
        for i in range(len(train)):
            for j in range(i+1, len(train)):
                print("Training LIBSVM for classes: {0} and {1}".format(i, j))
                X_train, y_train, X_test, y_test = self.build_input(class_index_0 = i, class_index_1= j)
                self.fit_svc(X_train, y_train, X_test, y_test, kernel)

    def analyse_sgd(self):
        global train, test
        for i in range(len(train)):
            for j in range(i+1, len(train)):
                print("Training LIBSVM for classes: {0} and {1}".format(i, j))
                X_train, y_train, X_test, y_test = self.build_input(class_index_0 = i, class_index_1= j)
                self.fit_sgd(X_train, y_train, X_test, y_test)
                # self.sort_and_plot(X_train)
    
    """
    1. Functional utilities for construction matrices for cvxopt solver
    """
    def build_matrix_linear_kernel(self, X, y, C):
        P = self.build_matrix_P(X, y)
        G, h = self.build_matrix_G_h(X, y, C)
        q, A, b = self.build_matrix_rest(X, y)
        return P, G, h, q, A, b

    def build_matrix_P(self, X, y):     
        p = np.multiply(X.T, y)
        P = np.matmul(p.T, p)
        print("Shape of P: ", P.shape)
        P = cvxopt.matrix(P)
        return P

    def build_matrix_G_h(self, X, y, C):
        m = X.shape[0]
        g1 = -1*np.diag(np.ones(m))
        g2 = np.identity(m)
        G = np.vstack((g1, g2))
        G = cvxopt.matrix(G)
        h1 = np.zeros(m)
        h2 = C*np.ones(m)
        h = np.hstack((h1, h2))
        h = cvxopt.matrix(h)
        return G, h
    
    def build_matrix_rest(self, X, y):
        m = X.shape[0]
        q = -1*np.ones(m)
        A = y.reshape((1,m))
        b = np.zeros((1,1))
        q = cvxopt.matrix(q)
        A = cvxopt.matrix(A)
        b = cvxopt.matrix(b)
        return q, A, b

    def build_matrix_gaussian_kernel(self, X, y, C, gamma):
        P = self.build_gaussian_P(X, y, gamma)
        G, h = self.build_matrix_G_h(X, y, C)
        q, A, b = self.build_matrix_rest(X, y)
        return P, G, h, q, A, b
    
    def build_gaussian_P(self, X, y, gamma):
        K = self.guassian(X, X, gamma)
        P = np.outer(y, y) * K
        return cvxopt.matrix(P)


    def guassian(self, X1, X2, gamma):
        X1_sq = np.einsum('ij,ij->i', X1, X1)[:, np.newaxis]
        X2_sq = np.einsum('ij,ij->i', X2, X2)[np.newaxis, :]
        sq_dists = X1_sq + X2_sq - 2 * np.einsum('ik,jk->ij', X1, X2)
        return np.exp(-gamma * sq_dists)


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

        tol = 1e-5

        if kernel == 'linear':
            P, G, h, q, A, b_ = self.build_matrix_linear_kernel(X, y, C)
        if kernel == 'gaussian':
            P, G, h, q, A, b_ = self.build_matrix_gaussian_kernel(X, y, C, gamma)
        solution = cvxopt.solvers.qp(P, q, G, h, A, b_,  options={'show_progress': False})
        alphas = np.array(solution['x']).reshape((len(solution['x']), ))
        global support_vectors, X_supp, y_supp
        support_indices = [i for i in range(len(alphas)) if alphas[i] > tol]
        support_vectors = alphas[support_indices]
        X_supp = X[support_indices]
        y_supp = y[support_indices]
        print("No. of support vectors: ",len(support_vectors))
        print("Percentage of support vectors of training examples: {} %".format(float(100*len(support_vectors))/X.shape[0]))

        global w, b
        if kernel == 'linear':
            w = np.dot(X_supp.T, np.multiply(support_vectors, y_supp))
            print("Shape of Coeff: ", w.shape)   
            print("Coeff: ", w)
            w_b = np.matmul(X_supp, w)
            indices_pos = [i for i in range(len(y_supp)) if y_supp[i] == 1]
            indices_neg = [i for i in range(len(y_supp)) if y_supp[i] == -1]
            w_b_T_pos = w_b[indices_pos]
            w_b_T_neg = w_b[indices_neg] 
            max = np.max(w_b_T_neg)
            min = np.min(w_b_T_pos)
            b = -1*(max + min)/2
            print("Intercept: ", b)  
        if kernel == 'gaussian':
            wxt = np.sum((support_vectors * y_supp).reshape((1, len(y_supp))) * self.guassian(X_supp, X_supp, gamma), axis = 0)
            b = np.mean(y_supp - wxt)
            print("Intercept: ", b)


    def fit_svc(self, X, y, X_, y_, kernel = 'linear', C = 1.0, gamma = 0.001):
        model = svm.SVC(C=C, kernel=kernel, gamma=gamma)
        param = model.fit(X,y)
        print(model.n_support_)
        support_indices = model.support_
        alpha_y = model.dual_coef_
        # print("Weights: ", model.coef_)
        print("Intercept: ", model.intercept_)
        print("Total support vectors:", sum(model.n_support_))
        y_pred = model.predict(X_)
        accuracy = self.accuracy(y_, y_pred)
        svc_accuracy = accuracy_score(y_, y_pred)
        print("Accuracy: ", accuracy, svc_accuracy)
        # self.sort_and_plot(X, model.coef_)

    def fit_sgd(self, X, y, X_, y_):
        model = SGDClassifier()
        param = model.fit(X,y)
        y_pred = model.predict(X_)
        accuracy = self.accuracy(y_, y_pred)
        sgd_accuracy = model.score(X_, y_)
        print("Accuracy: ", accuracy, sgd_accuracy)
        # self.sort_and_plot(X, model.coef_)


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
        global w, b
        print("b: ", b)
        distance = np.matmul(X, w) + b
        prediction = np.sign(distance)
        return prediction
    
    def predict_gaussian(self, X, gamma=0.001):
        global support_vectors, X_supp, y_supp, b
        wTx = np.sum((support_vectors * y_supp).reshape((len(y_supp), 1)) * self.guassian(X_supp, X, gamma), axis = 0)
        wTx += b
        prediction = np.sign(wTx)
        return prediction

    def accuracy(self, y, prediction):
        miss = 0
        for i in range(len(y)):
            if y[i] != prediction[i]:
                miss += 1
        accuracy = float(len(y) - miss)/len(y)
        print("Misses: ", miss)
        print("Accuracy: ", accuracy)
        return accuracy
    
    def sort_and_plot(self, X, w):
        index = np.argsort(w.reshape((30000,)))[-5:][::-1]
        print(index)
        for i in range(len(index)):
            X_ = X[:,i]      
            X_ = X_.reshape((100, 100, 3))
            cv2.imshow("Image", X_)
            cv2.waitKey(0)
        w_ = w.reshape((100, 100, 3))
        w_ = np.multiply(w_, 255)
        cv2.imshow("Image", w_)
        cv2.waitKey(0)

    def predict_multiclass(self):
        X, y = self.build_complete_test_input()
        votes = np.zeros((len(train), X.shape[0]))
        for i in range(len(train)):
            for j in range(i+1, len(train)):
                print("Training for classes: {0} and {1}".format(i, j))
                X_train, y_train, garbage, garbage_  = self.build_input(class_index_0 = i, class_index_1= j)
                self.fit(X_train, y_train, kernel='gaussian')
                prediction = self.predict_gaussian(X)
                print(prediction.shape)
                for k in range(X.shape[0]):
                    if prediction[k] == 1:
                        votes[j][k] += 1
                    else:
                        votes[i][k] += 1
        predicted_class = np.argmax(votes, axis = 0)
        accuracy = self.accuracy(y, predicted_class)
        print("Accuracy multi-class:", accuracy)
        print("Drawing confusion matrix...")
        cm = confusion_matrix(y, predicted_class)
        print(cm)
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y, predicted_class))
        return predicted_class

    def predict_multiclass_gaussian(self):
        print("Building data...")
        X_train, y_train = self.build_complete_train_input()
        X_test, y_test = self.build_complete_test_input()
        print("Training model...")
        model = svm.SVC(C=1.0, kernel='rbf', gamma=0.001, verbose=True)
        param = model.fit(X_train,y_train)
        print("Predicting...")
        prediction = model.predict(X_test)
        accuracy = self.accuracy(y_test, prediction)
        print("Accuracy multi-class gaussian:", accuracy)
        print("Drawing confusion matrix...")
        cm = confusion_matrix(y_test, prediction)
        print(cm)
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, prediction))
        return prediction

    def build_complete_test_input(self):
        global test
        X = test[0]
        y = np.zeros(test[0].shape[0])
        for i in range(len(test)):
            X = np.vstack((X, test[i]))
            y = np.hstack((y, i*np.ones(test[i].shape[0])))
        return X, y
    
    def build_complete_train_input(self):
        global train
        X = train[0]
        y = np.zeros(train[0].shape[0])
        for i in range(len(train)):
            X = np.vstack((X, train[i]))
            y = np.hstack((y, i*np.ones(train[i].shape[0])))
        return X, y
    
    def cross_validation_for_c(self, c, X, y, X_test, y_test):
        model = svm.SVC(C=c, kernel='rbf', gamma=0.001, verbose=True)
        cross_val_acc = np.mean(cross_val_score(model, X, y, cv = 5, verbose = 1000))
        model.fit(X, y)
        y_predicted = model.predict(X_test)
        test_acc = self.accuracy(y_test, y_predicted)
        
        print("cross validation accuracies for {}: ".format(c), cross_val_acc)
        print("test accuracies for {}".format(c), test_acc)
        return cross_val_acc, test_acc