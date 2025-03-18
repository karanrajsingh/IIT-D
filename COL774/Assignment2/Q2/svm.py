import cvxopt
import numpy as np
import time


class SupportVectorMachine:
    '''
    Binary Classifier using Support Vector Machine
    '''
    
    w = ""
    b = ""
    support_vectors = ""
    support_indices = ""
    X_supp = ""
    y_supp = ""
    model_kernel = ""
    model_gamma = ""

    def __init__(self):
        ""

    """
    TRAINING UTILITIES FOR CVXOPT SOLVER
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
        y = y.reshape((y.shape[0], 1))
        P = np.matmul(y, y.T) * K
        return cvxopt.matrix(P)

    def guassian(self, X1, X2, gamma):
        X1_sq = np.einsum('ij,ij->i', X1, X1)[:, np.newaxis]
        X2_sq = np.einsum('ij,ij->i', X2, X2)[np.newaxis, :]
        sq_dists = X1_sq + X2_sq - 2 * np.einsum('ik,jk->ij', X1, X2)
        return np.exp(-gamma * sq_dists)


    """
    TRAINING CVXOPT SOLVER FOR LINEAR AND GAUSSIAN KERNELS
    """

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

        # changing  y_train as 0/1 to -1/1
        y = 2*y - 1

        global model_kernel, model_gamma
        model_kernel = kernel
        model_gamma = gamma

        tol = 1e-5

        start = time.time()

        if kernel == 'linear':
            P, G, h, q, A, b_ = self.build_matrix_linear_kernel(X, y, C)
        if kernel == 'gaussian':
            P, G, h, q, A, b_ = self.build_matrix_gaussian_kernel(X, y, C, gamma)
        solution = cvxopt.solvers.qp(P, q, G, h, A, b_,  options={'show_progress': False})
        end = time.time()
        print("Training time for cxvopt: ", end - start)
        alphas = np.array(solution['x']).reshape((len(solution['x']), ))
        global support_vectors, support_indices, X_supp, y_supp
        support_indices = [i for i in range(len(alphas)) if alphas[i] > tol]
        support_vectors = alphas[support_indices]
        X_supp = X[support_indices]
        y_supp = y[support_indices]
        print("No. of support vectors: ",len(support_vectors))
        print("Percentage of support vectors of training examples: {} %".format(float(100*len(support_vectors))/X.shape[0]))
        # print("Support vector indices: ", support_indices)

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


    """"
    PREDICTION FUNCTIONS FOR CVXOPT
    """

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
        global model_kernel, model_gamma, b
        print("b: ", b)
        if model_kernel == 'linear':
            global w
            distance = np.matmul(X, w) + b
            prediction = np.sign(distance)
        else:
            global support_vectors, X_supp, y_supp
            wTx = np.sum((support_vectors * y_supp).reshape((len(y_supp), 1)) * self.guassian(X_supp, X, gamma = model_gamma), axis = 0)
            wTx += b
            prediction = np.sign(wTx)
        prediction = (prediction + 1)/2
        return prediction
    
    def predict_gaussian(self, X, gamma=0.001):
        global support_vectors, X_supp, y_supp, b
        wTx = np.sum((support_vectors * y_supp).reshape((len(y_supp), 1)) * self.guassian(X_supp, X, gamma), axis = 0)
        wTx += b
        prediction = np.sign(wTx)
        return prediction

    # Utilities for analysis
    def accuracy(self, y, prediction):
        miss = 0
        for i in range(len(y)):
            if y[i] != prediction[i]:
                miss += 1
        accuracy = float(len(y) - miss)/len(y)
        print("Misses: ", miss)
        print("Accuracy: ", accuracy)
        return accuracy
    
    def get_params(self):
        global model_kernel, w, b, support_vectors, X_supp, support_indices
        if model_kernel == 'linear':
            return w, b, support_vectors, X_supp, support_indices
        else:
            return support_vectors, X_supp, support_indices