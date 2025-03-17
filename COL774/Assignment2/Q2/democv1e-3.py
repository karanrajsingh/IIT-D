import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import svm

model = svm.SupportVectorMachine()
path = "../data/Q2/"
model.read_images(path)

X, y = model.build_complete_train_input()
X_test, y_test = model.build_complete_test_input()
n = X.shape[1]
Xy = np.concatenate((X, y.reshape((y.shape[0], 1))), axis=1)
np.random.shuffle(Xy)
X, y = Xy[:, 0 : n], Xy[:, n]


model.cross_validation_for_c(1e-3, X, y, X_test, y_test)

# plt.figure()
# plt.semilogx(C, cross_val_acc, marker='o', label='Cross Validation Accuracy')
# plt.semilogx(C, test_acc, marker='s', label='Test Accuracy')
# plt.xlabel("C")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()

# C_max = C[np.argmax(cross_val_acc)]
# model_best = svm.SVC(C=C_max, kernel='rbf', gamma=0.001, verbose=True)
# y_predicted = model_best.fit(X_test)
# print("Accuracy for best model: ", model.accuracy(y_test, y_predicted))