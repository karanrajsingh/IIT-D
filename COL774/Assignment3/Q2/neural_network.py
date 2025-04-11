import helper
import numpy as np

path = "../data/Q2"
X_train, y_train = helper.read_images_train(path)
X_test, y_test = helper.read_images_test(path)
print(y_train)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(y_test)
# print(array_list_test)