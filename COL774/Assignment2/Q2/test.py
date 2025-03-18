import svm as s
import helper
import numpy as np
from sklearn import svm

svm_model = s.SupportVectorMachine()

path = "../data/Q2/"
array_list_train, array_list_test = helper.read_images(path)

# X_train, y_train, X_test, y_test = helper.build_input(class_index_0 = 0, class_index_1= 1)

X, y = helper.build_complete_train_input(array_list_train)
X_test, y_test = helper.build_complete_test_input(array_list_test)

svm_model =svm.SVC(C=10, kernel='rbf', gamma=0.001, verbose=True)
svm_model.fit(X, y)
y_predicted = svm_model.predict(X)
train_acc = svm_model.accuracy(y, y_predicted)
print("train acc: ", train_acc)
y_predicted = svm_model.predict(X_test)
test_acc = svm_model.accuracy(y_test, y_predicted)
print("test acc: ", test_acc)

