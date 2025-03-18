import svm
import helper
import numpy as np

model = svm.SupportVectorMachine()

path = "../data/Q2/"
array_list_train, array_list_test = helper.read_images(path)

X_train, y_train, X_test, y_test = helper.build_input(class_index_0 = 0, class_index_1= 1)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

y = y_train.reshape((y_train.shape[0], 1))

A = np.matmul(y, y.T)
print(A.shape)
