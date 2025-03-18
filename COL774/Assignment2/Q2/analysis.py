import svm as s
import helper
import numpy as np
import cv2
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import time

svm_model = s.SupportVectorMachine()

"""
TRAIN and TEST SVC, SGD and LinearSVC FOR BINARY CASES
"""

def fit_svc(X, y, X_, y_, kernel = 'linear', C = 1.0, gamma = 0.001):
    model = svm.SVC(C=C, kernel=kernel, gamma=gamma)
    start = time.time()
    param = model.fit(X,y)
    end = time.time()
    print(model.n_support_)
    support_indices = model.support_
    # alpha_y = model.dual_coef_
    # print("Weights: ", model.coef_)
    # print("Intercept: ", model.intercept_)
    print("Total support vectors:", sum(model.n_support_))
    # print("Support vector indices: ", support_indices)
    y_pred = model.predict(X_)
    accuracy = svm_model.accuracy(y_, y_pred)
    svc_accuracy = accuracy_score(y_, y_pred)
    print("Accuracy: ", accuracy, svc_accuracy)
    print("Training time for {} case".format(kernel), end - start)
    # sort_and_plot(X, model.coef_)

def fit_sgd(X, y, X_, y_):
    model = SGDClassifier()
    start = time.time()
    param = model.fit(X,y)
    end = time.time()
    y_pred = model.predict(X_)
    accuracy = svm_model.accuracy(y_, y_pred)
    sgd_accuracy = model.score(X_, y_)
    print("Accuracy for fit sgd: ", accuracy, sgd_accuracy)
    print("Training time: ", end - start)
    # sort_and_plot(X, model.coef_)

def fit_liblinear(X, y, X_, y_, C = 1.0):
    model = svm.LinearSVC(C=C)
    start = time.time()
    param = model.fit(X,y)
    end = time.time()
    # print(model.n_support_)
    # support_indices = model.support_
    # alpha_y = model.dual_coef_
    print("Weights: ", model.coef_)
    print("Intercept: ", model.intercept_)
    # print("Total support vectors:", sum(model.n_support_))
    y_pred = model.predict(X_)
    accuracy = svm_model.accuracy(y_, y_pred)
    svc_accuracy = accuracy_score(y_, y_pred)
    print("Accuracy: ", accuracy, svc_accuracy)
    print("Training time for LinearSVC case", end - start)
    # sort_and_plot(X, model.coef_)


def sort_and_plot(X, w):
    index = np.argsort(w.reshape((30000,)))[-5:][::-1]
    print(index)
    for i in range(len(index)):
        X_ = X[i, :]      
        X_ = X_.reshape((100, 100, 3))
        cv2.imshow("Image", X_)
        cv2.waitKey(0)
    w_ = w.reshape((100, 100, 3))
    w_ = np.multiply(w_, 255)
    cv2.imshow("Image", w_)
    cv2.waitKey(0)


"""
ANALYSIS: MULTICLASS CVXOPT AND SVC FOR GAUSSIAN KERNEL
"""

def predict_multiclass(array_list_train, array_list_test):
    X, y= helper.build_complete_test_input(array_list_test)
    l = len(array_list_train)
    votes = np.zeros((l, X.shape[0]))
    times = []
    for i in range(l):
        for j in range(i+1, l):
            print("Training for classes: {0} and {1}".format(i, j))
            X_train, y_train, garbage, garbage_  = helper.build_input(array_list_train, array_list_test, class_index_0 = i, class_index_1= j)
            y_train = (1+y_train)/2
            start_train = time.time()
            svm_model.fit(X_train, y_train, kernel='gaussian')
            end_train = time.time()
            times.append(end_train - start_train)
            prediction = svm_model.predict_gaussian(X)
            for k in range(X.shape[0]):
                if prediction[k] == 1:
                    votes[j][k] += 1
                else:
                    votes[i][k] += 1
    predicted_class = np.argmax(votes, axis = 0)
    accuracy = svm_model.accuracy(y, predicted_class)
    print("Accuracy multi-class:", accuracy)
    cm = confusion_matrix(y, predicted_class)
    print(cm)
    print(classification_report(y, predicted_class))
    for i in range(len(y)):
        if y[i] != predicted_class[i]:
            print("Example {0}, {1}, {2}".format(i, y[i], predicted_class[i]))
    total_time_train = 0
    for i in range(len(times)):
        total_time_train += times[i]
    print("Training time for cvxopt: ", total_time_train)
    return predicted_class

# Analysis functions for MULTICLASS SVC
def predict_multiclass_gaussian(array_list_train, array_list_test):
    print("Building data...")
    X_train, y_train = helper.build_complete_train_input(array_list_train)
    X_test, y_test = helper.build_complete_test_input(array_list_test)
    print("Training model...")
    start_train = time.time()
    model = svm.SVC(C=1.0, kernel='rbf', gamma=0.001, verbose=True)
    param = model.fit(X_train,y_train)
    end_train = time.time()
    print("Predicting...")
    prediction = model.predict(X_test)
    accuracy = svm_model.accuracy(y_test, prediction)
    print("Accuracy multi-class gaussian:", accuracy)
    print("Drawing confusion matrix...")
    cm = confusion_matrix(y_test, prediction)
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, prediction))
    print("Training time for LIBSVM: ", end_train - start_train)
    for i in range(len(y_test)):
        if y_test[i] != prediction[i]:
            print("Example {0}, {1}, {2}".format(i, y_test[i], prediction[i]))
    return prediction


# Cross validation over MULTICLASS SVC
def cross_validation_for_c(c, X, y, X_test, y_test):
    model = svm.SVC(C=c, kernel='rbf', gamma=0.001, verbose=True)
    cross_val_acc = np.mean(cross_val_score(model, X, y, cv = 5, verbose = 1000))
    model.fit(X, y)
    y_predicted = model.predict(X_test)
    test_acc = svm_model.accuracy(y_test, y_predicted)
    
    print("cross validation accuracies for {}: ".format(c), cross_val_acc)
    print("test accuracies for {}".format(c), test_acc)
    return cross_val_acc, test_acc