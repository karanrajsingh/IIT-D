import numpy as np
from sklearn import svm
import svm
import helper
import analysis

model = svm.SupportVectorMachine()
path = "../data/Q2/"
array_list_train, array_list_test = helper.read_images(path)

X, y = helper.build_complete_train_input(array_list_train)
X_test, y_test = helper.build_complete_test_input(array_list_test)
n = X.shape[1]
Xy = np.concatenate((X, y.reshape((y.shape[0], 1))), axis=1)
np.random.shuffle(Xy)
X, y = Xy[:, 0 : n], Xy[:, n]

analysis.cross_validation_for_c(10, X, y, X_test, y_test)