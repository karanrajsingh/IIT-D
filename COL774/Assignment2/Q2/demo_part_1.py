import svm
import time

model = svm.SupportVectorMachine()

path = "../data/Q2/"
model.read_images(path)

X_train, y_train, X_test, y_test = model.build_input(class_index_0 = 0, class_index_1= 1)
y_train_adj = (y_train + 1)/2
y_test_adj =  (y_test + 1)/2

# Part 4
print("LIBLINEAR")
model.fit_liblinear(X_train, y_train, X_test, y_test, C = 1.0)
print("LIBLINEAR ends...")
print("SGD")
model.fit_sgd(X_train, y_train, X_test, y_test)
print("SGD ends...")

# Part 3 linear case, Part 1 (uncomment sort_and_plot in fit)
print("CVXOPT LINEAR")
model.fit(X_train, y_train_adj, kernel='linear')
y_predicted = model.predict(X_test)
accuracy = model.accuracy(y_test_adj, y_predicted)
print("Accuracy: ", accuracy)
print("CVXOPT LINEAR ends...")

# Part 3 linear case (uncomment weights and intercept inside fit_svc)
print("SVC LINEAR")
model.fit_svc(X_train, y_train, X_test, y_test, kernel='linear')
print("SVC LINEAR ends...")

# Part 3 gaussian case, Part 2 (uncomment sort_and_plot in fit)
print("CXVOPT GAUSSIAN")

model.fit(X_train, y_train_adj, kernel='gaussian')
y_predicted = model.predict(X_test)
accuracy = model.accuracy(y_test_adj, y_predicted)
print("Accuracy: ", accuracy)

# Part 3 gaussian case
print("SVC GAUSSIAN")
model.fit_svc(X_train, y_train, X_test, y_test, kernel='rbf')
