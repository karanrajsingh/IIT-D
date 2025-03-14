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
    w = ""
    b = ""
    

    def __init__(self):
        self.read_images("../data/Q2/")
        X_train, y_train, X_test, y_test = self.build_input()
        w, b = self.fit(X_train, y_train)
        prediction = self.predict(X_test)
        accuracy = self.accuracy(y_test, prediction)
        self.sort_and_plot(X_train)


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
    
    def build_matrix_linear_kernel(self, X, y, C = 1.0):
        P = self.build_matrix_P(X, y)
        G, h = self.build_matrix_G_h(X, y, C)
        q, A, b = self.build_matrix_rest(X, y)
        return P, G, h, q, A, b

    def build_input(self, class_index_0 = 0, class_index_1 = 1):
        global train
        X_train_0 = train[class_index_0]
        X_train_1 = train[class_index_1]
        X_train = np.hstack((X_train_0, X_train_1))
        m0 = X_train_0.shape[1]
        m1 = X_train_1.shape[1]
        y_train_0 = -1*np.ones(m0)
        y_train_1 = np.ones(m1)
        y_train = np.hstack((y_train_0, y_train_1))
        # print(X.shape)
        # print(y.shape)
        global test
        X_test_0 = test[class_index_0]
        X_test_1 = test[class_index_1]
        X_test = np.hstack((X_test_0, X_test_1))
        m0 = X_test_0.shape[1]
        m1 = X_test_1.shape[1]
        y_test_0 = -1*np.ones(m0)
        y_test_1 = np.ones(m1)
        y_test = np.hstack((y_test_0, y_test_1))
        return X_train, y_train, X_test, y_test 

    def build_matrix_P(self, X, y):
        # global train
        # X_train_0 = train[class_index_0]
        # X_train_1 = train[class_index_1]
        # yX_train_0 = -1*X_train_0
        # yX_train_1 = 1*X_train_1

        # p = np.hstack((yX_train_0, yX_train_1))        
        p = np.multiply(X, y)
        P = np.matmul(p.T, p)
        P = cvxopt.matrix(P)
        return P

    def build_matrix_G_h(self, X, y, C = 1.0):
        # global train
        # X_train_0 = train[class_index_0]
        # X_train_1 = train[class_index_1]
        # m = X_train_0.shape[1] + X_train_1.shape[1]
        m = X.shape[1]
        g1 = -1*np.diag(np.ones(m))
        g2 = np.identity(m)
        G = np.vstack((g1, g2))
        G = cvxopt.matrix(G)
        h1 = np.zeros(m)
        h2 = C*np.ones(m)
        h = np.hstack((h1, h2))
        # print(h.shape)
        h = cvxopt.matrix(h)
        return G, h
    
    def build_matrix_rest(self, X, y):
        # global train
        # X_train_0 = train[class_index_0]
        # X_train_1 = train[class_index_1]
        # m = X_train_0.shape[1] + X_train_1.shape[1]
        m = X.shape[1]
        q = -1*np.ones(m)
        # a1 = -1*np.ones(X_train_0.shape[1])
        # a2 = np.ones(X_train_1.shape[1])
        # A = np.hstack((a1, a2)).reshape((1,m))
        A = y.reshape((1,m))
        # print(A.shape)
        b = np.zeros((1,1))
        # print(b)
        q = cvxopt.matrix(q)
        A = cvxopt.matrix(A)
        b = cvxopt.matrix(b)
        return q, A, b

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

        P, G, h, q, A, b_ = self.build_matrix_linear_kernel(X, y, C)

        solution = cvxopt.solvers.qp(P, q, G, h, A, b_)
        alphas = np.array(solution['x']).reshape((len(solution['x']), ))
        # print(alphas)
        support_vectors = [alpha for alpha in alphas  if alpha > 1e-5]
        print("No. of support vectors: ",len(support_vectors))
        print("Percentage of support vectors of training examples: {} %".format(float(100*len(support_vectors))/X.shape[1]))

        global w, b
        w = np.dot(X, np.multiply(alphas, y))
        '''
        b = 0
        y_i(w.T*x_i + b) = 1, for support vectors
        if y_i == -1
            (w.T*x_i + b) = -1
            b += -1 - w.T*x_i
        if y_i == 1
            (w.T*x_i + b) = 1
            b += 1 - w.T*x_i
        b = b/num_supp_vec       
        '''       
        b = 0
        for i in range(len(alphas)):
            if alphas[i] > 1e-5:
                if y[i] > 0:
                    b += -1 - np.matmul(w.T, X[:,i])
                else:
                    b += 1 - np.matmul(w.T, X[:,i])                    
            b = float(b)/len(support_vectors)
        print("Weight vector: ", w)
        print("Intercept: ", b)
        return w, b

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
        distance = np.matmul(w.T, X) + b
        prediction = np.sign(distance)
        return prediction
    
    def accuracy(self, y, prediction):
        miss = 0
        print(y.shape)
        print(prediction.shape)
        for i in range(len(y)):
            if y[i] != prediction[i]:
                miss += 1
        accuracy = float(len(y) - miss)/len(y)
        print("Misses: ", miss)
        print("Accuracy: ", accuracy)
        return accuracy
    
    def sort_and_plot(self, X):
        global w
        index = np.argsort(w)[-5:]
        for i in range(len(index)):
            X_ = X[:,i]
            # print(X_.shape)         
            X_ = X_.reshape((100, 100, 3))
            # print(X_.shape)
            cv2.imshow("Image", X_)
            cv2.waitKey(0)
        w_ = w.reshape((100, 100, 3))
        w_ = np.multiply(w_, 255)
        cv2.imshow("Image", w_)
        cv2.waitKey(0)