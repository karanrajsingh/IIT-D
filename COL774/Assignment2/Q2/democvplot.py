import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import svm

# model = svm.SupportVectorMachine()
# path = "../data/Q2/"
# model.read_images(path)


cross_val_acc = []
test_acc = []
C = [1e-5, 1e-3, 1.0, 5.0, 10.0]
cross_val_acc = [0.18473767384790668, 0.184737536905877, 0.6983913419771142, 0.7149459900634862, 0.7187531154311756]
test_acc = [0.1845748187211602, 0.1845748187211602, 0.6677653263019117, 0.6875411997363217, 0.6908371786420567]

plt.figure()
plt.semilogx(C, cross_val_acc, marker='o', label='Cross Validation Accuracy')
plt.semilogx(C, test_acc, marker='s', label='Test Accuracy')
plt.xlabel("C")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# C_max = C[np.argmax(cross_val_acc)]
# model_best = svm.SVC(C=C_max, kernel='rbf', gamma=0.001, verbose=True)
# y_predicted = model_best.fit(X_test)
# print("Accuracy for best model: ", model.accuracy(y_test, y_predicted))